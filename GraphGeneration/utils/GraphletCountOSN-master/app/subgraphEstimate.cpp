#include "IMPR.h"
#include "PSRW.h"
#include "errorMetric.h"
#include "graphlet.h"
#include "process.h"
#include "utility.h"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using std::string;
using std::vector;

static void print_vec(const vector<double>& v) {
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) std::cout << ' ';
    std::cout << std::setprecision(10) << v[i];
  }
  std::cout << '\n';
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " graphfile size-k [steps] [trials]\n";
    return 1;
  }
  const char* graphfile = argv[1];
  int k      = std::atoi(argv[2]);
  int steps  = (argc > 3) ? std::atoi(argv[3]) : 10000;   // default walk length
  int trials = (argc > 4) ? std::atoi(argv[4]) : 100;     // default repeats

  int n, m;
  vector<vector<int>> edges;
  readGraph(graphfile, edges, 3);
  preprocess(edges, n, m); // largest CC

  // Choose RW variants (same as original, but we won't log/compare)
  vector<string> rwTypes = (k == 3)
      ? vector<string>{"SRW", "SRWNOE"}
      : vector<string>{"SRW", "SRWIMPR", "SRWNOE", "SRWIMPRNOE"};

  // Storage for running means per RW
  size_t variants = rwTypes.size();
  vector<vector<double>> mean_est(variants);  // will size after first run

  for (int t = 0; t < trials; ++t) {
    vector<vector<double>> estimates;
    if (k == 3) {
      estimates = node3Count_B(edges, steps, n, m);
    } else if (k == 4) {
      estimates = node4Count_B(edges, steps, n, m);
    } else if (k == 5) {
      estimates = node5Count_B(edges, steps, n, m);
    } else {
      std::cerr << "Only k=3,4,5 supported.\n";
      return 2;
    }
    if (t == 0) {
      mean_est.resize(estimates.size());
      for (size_t v = 0; v < estimates.size(); ++v)
        mean_est[v] = vector<double>(estimates[v].size(), 0.0);
    }
    for (size_t v = 0; v < estimates.size(); ++v) {
      for (size_t i = 0; i < estimates[v].size(); ++i) {
        // online mean
        mean_est[v][i] += (estimates[v][i] - mean_est[v][i]) / double(t + 1);
      }
    }
  }

  // Output: one line per RW variant, space-separated estimates
  // Prefix with the RW name so you can parse which line is which.
  for (size_t v = 0; v < variants; ++v) {
    std::cout << rwTypes[v] << ' ';
    print_vec(mean_est[v]);
  }
  return 0;
}
