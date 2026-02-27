/**
 * wl_stochtree_api.cpp -- C glue layer for stochtree WASM port.
 *
 * Wraps stochtree's C++ classes in extern "C" functions for Emscripten.
 * Uses a high-level approach: the MCMC training loop runs entirely in C++.
 * JS calls wl_st_bart_create + wl_st_bart_fit + wl_st_bart_predict.
 */

#include <stochtree/container.h>
#include <stochtree/data.h>
#include <stochtree/ensemble.h>
#include <stochtree/leaf_model.h>
#include <stochtree/meta.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/tree_sampler.h>
#include <stochtree/variance_model.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

// Normal CDF: Phi(x) = 0.5 * erfc(-x / sqrt(2))
static inline double norm_cdf(double x) {
  return 0.5 * std::erfc(-x * M_SQRT1_2);
}

// Inverse normal CDF (probit function) using rational approximation
// Abramowitz & Stegun 26.2.23 / Peter Acklam's refinement
static double norm_ppf(double p) {
  if (p <= 0.0) return -1e30;
  if (p >= 1.0) return 1e30;

  // Rational approximation coefficients
  static const double a[] = {
    -3.969683028665376e+01, 2.209460984245205e+02,
    -2.759285104469687e+02, 1.383577518672690e+02,
    -3.066479806614716e+01, 2.506628277459239e+00
  };
  static const double b[] = {
    -5.447609879822406e+01, 1.615858368580409e+02,
    -1.556989798598866e+02, 6.680131188771972e+01,
    -1.328068155288572e+01
  };
  static const double c[] = {
    -7.784894002430293e-03, -3.223964580411365e-01,
    -2.400758277161838e+00, -2.549732539343734e+00,
     4.374664141464968e+00,  2.938163982698783e+00
  };
  static const double d[] = {
    7.784695709041462e-03, 3.224671290700398e-01,
    2.445134137142996e+00, 3.754408661907416e+00
  };

  double q, r;
  if (p < 0.02425) {
    // Lower tail
    q = std::sqrt(-2.0 * std::log(p));
    return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
  } else if (p <= 0.97575) {
    // Central
    q = p - 0.5;
    r = q * q;
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
  } else {
    // Upper tail
    q = std::sqrt(-2.0 * std::log(1.0 - p));
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
             ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
  }
}

// Thread-local error buffer
static char g_last_error[512] = "";

static void set_error(const char* msg) {
  std::strncpy(g_last_error, msg, sizeof(g_last_error) - 1);
  g_last_error[sizeof(g_last_error) - 1] = '\0';
}

// Internal model struct: owns all C++ state
struct WlBartModel {
  // Configuration
  int num_trees = 200;
  int num_gfr = 10;
  int num_burnin = 200;
  int num_samples = 100;
  double alpha = 0.95;
  double beta = 2.0;
  int min_samples_leaf = 5;
  int max_depth = -1;
  double leaf_scale = -1.0; // -1 = auto
  int cutpoint_grid = 100;
  int random_seed = 42;
  int task = 0; // 0 = regression, 1 = classification (probit)

  // Data standardization (regression)
  double y_bar = 0.0;
  double y_std = 1.0;
  int num_features = 0;

  // Posterior state
  std::unique_ptr<StochTree::ForestContainer> forest_container;
  std::vector<double> sigma2_samples;

  // Classification metadata
  int nr_class = 0;
  std::vector<int> classes;

  // Training labels for classification augmentation (not serialized)
  // These are only used during fit()
};

extern "C" {

const char* wl_st_get_last_error(void) {
  return g_last_error;
}

void* wl_st_bart_create(
    int num_trees, int num_gfr, int num_burnin, int num_samples,
    double alpha, double beta, int min_samples_leaf, int max_depth,
    double leaf_scale, int cutpoint_grid, int random_seed) {
  try {
    auto* m = new WlBartModel();
    m->num_trees = num_trees > 0 ? num_trees : 200;
    m->num_gfr = num_gfr >= 0 ? num_gfr : 10;
    m->num_burnin = num_burnin >= 0 ? num_burnin : 200;
    m->num_samples = num_samples > 0 ? num_samples : 100;
    m->alpha = alpha > 0 ? alpha : 0.95;
    m->beta = beta > 0 ? beta : 2.0;
    m->min_samples_leaf = min_samples_leaf > 0 ? min_samples_leaf : 5;
    m->max_depth = max_depth;
    m->leaf_scale = leaf_scale;
    m->cutpoint_grid = cutpoint_grid > 0 ? cutpoint_grid : 100;
    m->random_seed = random_seed;
    return static_cast<void*>(m);
  } catch (const std::exception& e) {
    set_error(e.what());
    return nullptr;
  }
}

int wl_st_bart_fit(
    void* handle,
    const double* X, int nrow, int ncol,
    const double* y,
    int task) {
  if (!handle) { set_error("null handle"); return -1; }
  auto* m = static_cast<WlBartModel*>(handle);
  m->task = task;
  m->num_features = ncol;

  try {
    // 1. Standardize y for regression; for classification, map to {-1, +1} for probit
    std::vector<double> y_std_vec(nrow);
    if (task == 0) {
      // Regression: standardize (y - mean) / sd
      double sum = 0.0;
      for (int i = 0; i < nrow; i++) sum += y[i];
      m->y_bar = sum / nrow;
      double sum_sq = 0.0;
      for (int i = 0; i < nrow; i++) {
        double d = y[i] - m->y_bar;
        sum_sq += d * d;
      }
      m->y_std = nrow > 1 ? std::sqrt(sum_sq / (nrow - 1)) : 1.0;
      if (m->y_std < 1e-12) m->y_std = 1.0;
      for (int i = 0; i < nrow; i++) {
        y_std_vec[i] = (y[i] - m->y_bar) / m->y_std;
      }
    } else {
      // Classification: Albert-Chib probit BART
      // Collect unique classes
      std::vector<int> unique_classes;
      for (int i = 0; i < nrow; i++) {
        int c = static_cast<int>(y[i]);
        bool found = false;
        for (int uc : unique_classes) {
          if (uc == c) { found = true; break; }
        }
        if (!found) unique_classes.push_back(c);
      }
      std::sort(unique_classes.begin(), unique_classes.end());
      m->nr_class = static_cast<int>(unique_classes.size());
      m->classes = unique_classes;

      // y_bar = Phi^{-1}(mean(y_binary)), y_std = 1.0
      // Map labels to binary {0, 1} for class indexing
      double y_mean = 0.0;
      for (int i = 0; i < nrow; i++) {
        y_mean += (static_cast<int>(y[i]) == unique_classes[1]) ? 1.0 : 0.0;
      }
      y_mean /= nrow;
      m->y_bar = norm_ppf(y_mean);
      m->y_std = 1.0;

      // Initial residuals: y_binary - mean(y_binary)
      for (int i = 0; i < nrow; i++) {
        double y_bin = (static_cast<int>(y[i]) == unique_classes[1]) ? 1.0 : 0.0;
        y_std_vec[i] = y_bin - y_mean;
      }
    }

    // 2. Create ForestDataset
    // stochtree expects row-major doubles
    StochTree::ForestDataset dataset;
    dataset.AddCovariates(const_cast<double*>(X), nrow, ncol, true);

    // 3. Create residual ColumnVector
    StochTree::ColumnVector residual(y_std_vec.data(), nrow);

    // 4. Create ForestContainer (stores posterior samples)
    m->forest_container = std::make_unique<StochTree::ForestContainer>(
        m->num_trees, 1, true, false);

    // 5. Create active forest (TreeEnsemble) -- single forest for iteration
    StochTree::TreeEnsemble active_forest(m->num_trees, 1, true, false);

    // 6. Feature types (all numeric for now)
    std::vector<StochTree::FeatureType> feature_types(ncol, StochTree::FeatureType::kNumeric);

    // 7. Create ForestTracker and TreePrior
    StochTree::ForestTracker tracker(dataset.GetCovariates(), feature_types,
                                     m->num_trees, nrow);
    StochTree::TreePrior split_prior(m->alpha, m->beta, m->min_samples_leaf, m->max_depth);

    // 8. RNG
    std::mt19937 rng(m->random_seed);

    // 9. Variance model
    StochTree::GlobalHomoskedasticVarianceModel var_model;
    double global_variance = 1.0;
    double a_var = 16.0; // IG shape prior
    double b_var = 4.0;  // IG scale prior

    // 10. Leaf model scale
    double leaf_sc = m->leaf_scale;
    if (leaf_sc <= 0) {
      // Auto: 1/num_trees is standard BART default
      leaf_sc = 1.0 / m->num_trees;
    }

    // 11. Variable weights (uniform)
    std::vector<double> var_weights(ncol, 1.0 / ncol);

    // 12. Sweep update indices (must list every tree index)
    std::vector<int> sweep_indices(m->num_trees);
    for (int i = 0; i < m->num_trees; i++) sweep_indices[i] = i;

    // 13. Initialize leaf values: y_bar_std / num_trees for constant leaf
    double y_bar_std = 0.0;
    for (int i = 0; i < nrow; i++) y_bar_std += y_std_vec[i];
    y_bar_std /= nrow;
    double init_leaf = y_bar_std / m->num_trees;
    active_forest.SetLeafValue(init_leaf);

    // Initialize tracker predictions
    StochTree::UpdateResidualEntireForest(
        tracker, dataset, residual, &active_forest, false, std::minus<double>());
    tracker.UpdatePredictions(&active_forest, dataset);

    // Create leaf model
    Eigen::MatrixXd leaf_scale_matrix; // unused for constant leaf
    StochTree::LeafModelVariant leaf_model = StochTree::leafModelFactory(
        StochTree::ModelType::kConstantLeafGaussian,
        leaf_sc, leaf_scale_matrix, a_var, b_var);

    // 14. For classification, store binary labels for data augmentation
    std::vector<int> y_binary(nrow, 0);
    if (task == 1) {
      for (int i = 0; i < nrow; i++) {
        y_binary[i] = (static_cast<int>(y[i]) == m->classes[1]) ? 1 : 0;
      }
    }
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    // 15. MCMC iterations: GFR warm-start + burn-in + sampling
    int total_iters = m->num_gfr + m->num_burnin + m->num_samples;
    m->sigma2_samples.clear();
    m->sigma2_samples.reserve(m->num_samples);

    for (int iter = 0; iter < total_iters; iter++) {
      bool gfr = (iter < m->num_gfr);
      bool keep_forest = (iter >= m->num_gfr + m->num_burnin);

      // Albert-Chib data augmentation for probit classification
      if (task == 1) {
        for (int i = 0; i < nrow; i++) {
          double mu = tracker.GetSamplePrediction(i);
          double u;
          if (y_binary[i] == 0) {
            // z ~ TruncNorm(-inf, 0 | mu, 1)
            double upper = norm_cdf(-mu);
            if (upper < 1e-15) upper = 1e-15;
            u = unif(rng) * upper;
          } else {
            // z ~ TruncNorm(0, inf | mu, 1)
            double lower = norm_cdf(-mu);
            if (lower > 1.0 - 1e-15) lower = 1.0 - 1e-15;
            u = lower + unif(rng) * (1.0 - lower);
          }
          double z = mu + norm_ppf(u);
          // Update residual: new_outcome = z - forest_prediction
          residual.SetElement(i, z - mu);
        }
      }

      if (gfr) {
        StochTree::GFRSampleOneIter<StochTree::GaussianConstantLeafModel,
                                     StochTree::GaussianConstantSuffStat>(
            active_forest, tracker, *(m->forest_container),
            std::get<StochTree::GaussianConstantLeafModel>(leaf_model),
            dataset, residual, split_prior, rng,
            var_weights, sweep_indices, global_variance,
            feature_types, m->cutpoint_grid,
            keep_forest, true, true, ncol, -1);
      } else {
        StochTree::MCMCSampleOneIter<StochTree::GaussianConstantLeafModel,
                                      StochTree::GaussianConstantSuffStat>(
            active_forest, tracker, *(m->forest_container),
            std::get<StochTree::GaussianConstantLeafModel>(leaf_model),
            dataset, residual, split_prior, rng,
            var_weights, sweep_indices, global_variance,
            keep_forest, true, true, -1);
      }

      // Sample global variance (regression only; probit fixes variance at 1)
      if (task == 0) {
        global_variance = var_model.SampleVarianceParameter(
            residual.GetData(), a_var, b_var, rng);
      }

      if (keep_forest) {
        m->sigma2_samples.push_back(global_variance);
      }
    }

    return 0;
  } catch (const std::exception& e) {
    set_error(e.what());
    return -1;
  }
}

int wl_st_bart_predict(
    void* handle,
    const double* X, int nrow, int ncol,
    double* out) {
  if (!handle) { set_error("null handle"); return -1; }
  auto* m = static_cast<WlBartModel*>(handle);
  if (!m->forest_container) { set_error("model not fitted"); return -1; }

  try {
    // Create dataset for prediction
    StochTree::ForestDataset dataset;
    dataset.AddCovariates(const_cast<double*>(X), nrow, ncol, true);

    // Predict: returns column-major vector (num_samples * nrow)
    std::vector<double> raw = m->forest_container->Predict(dataset);
    int ns = m->forest_container->NumSamples();

    // Average across posterior samples and de-standardize
    for (int i = 0; i < nrow; i++) {
      double sum = 0.0;
      for (int s = 0; s < ns; s++) {
        sum += raw[s * nrow + i];
      }
      double avg = sum / ns;
      if (m->task == 0) {
        // Regression: de-standardize
        out[i] = avg * m->y_std + m->y_bar;
      } else {
        // Classification: probit link (standard normal CDF)
        // Phi(z) approx using erfc
        out[i] = 0.5 * std::erfc(-avg / std::sqrt(2.0));
      }
    }

    return 0;
  } catch (const std::exception& e) {
    set_error(e.what());
    return -1;
  }
}

int wl_st_bart_predict_raw(
    void* handle,
    const double* X, int nrow, int ncol,
    double* out, int* out_num_samples) {
  if (!handle) { set_error("null handle"); return -1; }
  auto* m = static_cast<WlBartModel*>(handle);
  if (!m->forest_container) { set_error("model not fitted"); return -1; }

  try {
    StochTree::ForestDataset dataset;
    dataset.AddCovariates(const_cast<double*>(X), nrow, ncol, true);

    std::vector<double> raw = m->forest_container->Predict(dataset);
    int ns = m->forest_container->NumSamples();
    *out_num_samples = ns;

    // Output layout: row-major [nrow][ns]
    for (int i = 0; i < nrow; i++) {
      for (int s = 0; s < ns; s++) {
        double val = raw[s * nrow + i];
        if (m->task == 0) {
          out[i * ns + s] = val * m->y_std + m->y_bar;
        } else {
          out[i * ns + s] = 0.5 * std::erfc(-val / std::sqrt(2.0));
        }
      }
    }

    return 0;
  } catch (const std::exception& e) {
    set_error(e.what());
    return -1;
  }
}

char* wl_st_bart_to_json(void* handle) {
  if (!handle) { set_error("null handle"); return nullptr; }
  auto* m = static_cast<WlBartModel*>(handle);
  if (!m->forest_container) { set_error("model not fitted"); return nullptr; }

  try {
    // Build a JSON object containing forest + metadata
    nlohmann::json root;
    root["forest_container"] = m->forest_container->to_json();
    root["y_bar"] = m->y_bar;
    root["y_std"] = m->y_std;
    root["task"] = m->task;
    root["num_features"] = m->num_features;
    root["num_trees"] = m->num_trees;
    root["num_gfr"] = m->num_gfr;
    root["num_burnin"] = m->num_burnin;
    root["num_samples"] = m->num_samples;
    root["alpha"] = m->alpha;
    root["beta"] = m->beta;
    root["min_samples_leaf"] = m->min_samples_leaf;
    root["max_depth"] = m->max_depth;
    root["leaf_scale"] = m->leaf_scale;
    root["cutpoint_grid"] = m->cutpoint_grid;
    root["random_seed"] = m->random_seed;
    root["nr_class"] = m->nr_class;
    root["classes"] = m->classes;
    root["sigma2_samples"] = m->sigma2_samples;

    std::string json_str = root.dump();
    char* result = static_cast<char*>(std::malloc(json_str.size() + 1));
    if (!result) { set_error("malloc failed"); return nullptr; }
    std::memcpy(result, json_str.c_str(), json_str.size() + 1);
    return result;
  } catch (const std::exception& e) {
    set_error(e.what());
    return nullptr;
  }
}

void* wl_st_bart_from_json(const char* json_str) {
  if (!json_str) { set_error("null json string"); return nullptr; }

  try {
    nlohmann::json root = nlohmann::json::parse(json_str);

    auto* m = new WlBartModel();
    m->y_bar = root.value("y_bar", 0.0);
    m->y_std = root.value("y_std", 1.0);
    m->task = root.value("task", 0);
    m->num_features = root.value("num_features", 0);
    m->num_trees = root.value("num_trees", 200);
    m->num_gfr = root.value("num_gfr", 10);
    m->num_burnin = root.value("num_burnin", 200);
    m->num_samples = root.value("num_samples", 100);
    m->alpha = root.value("alpha", 0.95);
    m->beta = root.value("beta", 2.0);
    m->min_samples_leaf = root.value("min_samples_leaf", 5);
    m->max_depth = root.value("max_depth", -1);
    m->leaf_scale = root.value("leaf_scale", -1.0);
    m->cutpoint_grid = root.value("cutpoint_grid", 100);
    m->random_seed = root.value("random_seed", 42);
    m->nr_class = root.value("nr_class", 0);
    if (root.contains("classes")) {
      m->classes = root["classes"].get<std::vector<int>>();
    }
    if (root.contains("sigma2_samples")) {
      m->sigma2_samples = root["sigma2_samples"].get<std::vector<double>>();
    }

    // Reconstruct ForestContainer from JSON
    m->forest_container = std::make_unique<StochTree::ForestContainer>(
        m->num_trees, 1, true, false);
    if (root.contains("forest_container")) {
      m->forest_container->from_json(root["forest_container"]);
    }

    return static_cast<void*>(m);
  } catch (const std::exception& e) {
    set_error(e.what());
    return nullptr;
  }
}

void wl_st_free_string(char* str) {
  if (str) std::free(str);
}

void wl_st_bart_free(void* handle) {
  if (handle) {
    delete static_cast<WlBartModel*>(handle);
  }
}

int wl_st_bart_num_samples(void* handle) {
  if (!handle) return 0;
  auto* m = static_cast<WlBartModel*>(handle);
  if (!m->forest_container) return 0;
  return m->forest_container->NumSamples();
}

int wl_st_bart_num_trees(void* handle) {
  if (!handle) return 0;
  return static_cast<WlBartModel*>(handle)->num_trees;
}

int wl_st_bart_num_features(void* handle) {
  if (!handle) return 0;
  return static_cast<WlBartModel*>(handle)->num_features;
}

int wl_st_bart_get_sigma2(void* handle, double* out) {
  if (!handle) { set_error("null handle"); return -1; }
  auto* m = static_cast<WlBartModel*>(handle);
  for (size_t i = 0; i < m->sigma2_samples.size(); i++) {
    out[i] = m->sigma2_samples[i];
  }
  return static_cast<int>(m->sigma2_samples.size());
}

int wl_st_bart_get_task(void* handle) {
  if (!handle) return -1;
  return static_cast<WlBartModel*>(handle)->task;
}

int wl_st_bart_get_nr_class(void* handle) {
  if (!handle) return 0;
  return static_cast<WlBartModel*>(handle)->nr_class;
}

} // extern "C"
