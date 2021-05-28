#include "heif_reader.hpp"

#include <cmath>
#include <numbers>
#include <fmt/format.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/optim.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/utility.hpp>
#include <vector>

class FitCosine : public cv::ConjGradSolver::Function {
  public:
    FitCosine(std::vector<cv::Point2d> observations) {
        _observations = observations;
    }
    int getDims() const override { return 3; }

    double calc(const double* w) const override {
        double err = 0.0;
        for (const auto& p : _observations) {
            double diff = p.y - w[0] * std::cos(p.x * w[1] + w[2]);
            err += diff * diff;
        }

        return err / _observations.size();
    }

    void getGradient(const double* w, double* grad) override {
        double gradAmp = 0.0;
        double gradFreq = 0.0;
        double gradPhase = 0.0;

        for (const auto& p : _observations) {
            double cos = std::cos(p.x * w[1] + w[2]);
            double sin = std::sin(p.x * w[1] + w[2]);
            gradAmp += 2.0 * (w[0] * cos - p.y) * cos;
            gradFreq += 2.0 * (w[0] * cos - p.y) * w[0] * -sin * p.x;
            gradPhase += 2.0 * (w[0] * cos - p.y) * w[0] * -sin;
        }

        grad[0] = gradAmp / _observations.size();
        grad[1] = gradFreq / _observations.size();
        grad[2] = gradPhase / _observations.size();
    }

    void setObservations(std::vector<cv::Point2d> observations) {
        _observations = observations;
    }

  private:
    std::vector<cv::Point2d> _observations;
};

static std::vector<cv::Point2d>
generateCosineData(size_t num = 512,
                   double low = 0.0,
                   double high = 1.0,
                   double amp = 1.0,
                   double freq = 2.0 * std::numbers::pi,
                   double phase = 0.0,
                   double noiseStd = 0.0) {
    std::vector<cv::Point2d> points(num);
    auto rng = cv::theRNG();

    for (size_t i = 0; i < num; i++) {
        double x = rng.uniform(low, high);
        double y = amp * std::cos(freq * x + phase);
        points[i].x = x;
        points[i].y = y + rng.gaussian(noiseStd);
    }

    return points;
}

int main(int argc, char* argv[]) {
    constexpr double amp = 1.0;
    constexpr double freq = 2.0 * std::numbers::pi;
    constexpr double phase = -0.5 * std::numbers::pi;

    auto groundtruth = generateCosineData(128, 0.0, 1.0, amp, freq, phase, 0.0);
    auto observations =
        generateCosineData(128, 0.0, 1.0, amp, freq, phase, 0.2);

    auto func =
        cv::Ptr<cv::MinProblemSolver::Function>(new FitCosine(observations));

    auto solver = cv::ConjGradSolver::create(func);

    cv::Vec3d w = {1.0, 3.0, 1.0};
    double err = solver->minimize(w);
    fmt::print("Solved parameters: [{0:.2f}, {1:.2f}, {2:.2f}], Error: {3:.4f}\n", w[0], w[1], w[2], err);

    return 0;
}