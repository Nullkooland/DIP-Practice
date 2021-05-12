#include <opencv2/core.hpp>
#include <opencv2/core/optim.hpp>



int main(int argc, char* argv[]) {
    auto solver = cv::ConjGradSolver::create();
    return 0;
}