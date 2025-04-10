#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <chrono>
#include <climits>
#include "json.hpp"
#include <random>
#include <math.h>
#include <algorithm>
#include <Eigen/Dense>
 
using Eigen::MatrixXd;
using Eigen::VectorXd;

using namespace std;
using json = nlohmann::json;

#include <IpIpoptApplication.hpp>
#include <IpTNLP.hpp>
using namespace Ipopt;

class QuadraticNLP: public Ipopt::TNLP {
public:
    QuadraticNLP(const Eigen::VectorXd& B, const VectorXd& c): B_(B), c_(c), solution_(0.0) {}

    // Public getter
    double getSolution() const {
        return solution_;
    }

    bool get_nlp_info(Index& n, Index& m, Index& nnz_jac_g, Index& nnz_h_lag, IndexStyleEnum& index_style) override {
        // Only one variable: scalar u
        n = 1;
        // constraints
        m = 0;
        nnz_jac_g = 0;
        // 1 variable => Hessian is 1x1
        nnz_h_lag = 1;
        index_style = C_STYLE;
        return true;
    }

    bool get_bounds_info(Index n, Number* x_l, Number* x_u, Index m, Number *g_l, Number* g_u) override {
        //Let u range from -1e19 to +1e19
        x_l[0] = -10;
        x_u[0] = 10;
        return true;
    }

    bool get_starting_point(Index n, bool init_x, Number* x, bool init_z, Number* z_L, Number* z_U, Index m, bool init_lambda, Number *lambda) override {
        // Initial guess for u
        x[0] = 0.0;
        return true;
    }

    bool eval_f(Index n, const Number* x, bool new_x, Number& obj_value) override {
        // Cost function: (B*u - c) ^ 2
        double u = x[0];
        VectorXd diff = B_ * u - c_;
        obj_value = diff.squaredNorm();
        return true;
    }

    bool eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f) override {
        // Gradient of the cost function: 2 * B^T * B * u - 2 * B^T * c
        double u = x[0];

        // Convert eigen multiplaction to number
        VectorXd grad = 2.0 * (B_.transpose() * B_ * u - B_.transpose() * c_);
        grad_f[0] = grad(0);
        return true;
    }

    bool eval_g(Index n, const Number* x, bool new_x, Index m, Number* g) override {
        // No constraints
        return true;
    }

    bool eval_jac_g(Index n, const Number* x, bool new_x, Index m, Index nnz_jac_g, Index* iRow, Index* jCol, Number* values) override {
        // No constraints
        return true;
    }

    bool eval_h(Index n, const Number* x, bool new_x, Number obj_factor, Index m, const Number* lambda, bool new_lambda, Index nnz_h_lag, Index* iRow, Index* jCol, Number* values) override {
        // Hessian of the cost wrt u: 2 * B^T * B
        if (values == nullptr) {
            // Only one entry (row 0, col 0)
            iRow[0] = 0;
            jCol[0] = 0;
        }
        else {
            values[0] = obj_factor * 2.0 * B_.squaredNorm();
        }
        return true;
    }

    void finalize_solution(SolverReturn status, Index n, const Number* x, const Number* z_L, const Number* z_U, Index m, const Number* g, const Number* lambda, Number obj_value, const IpoptData* ip_data, IpoptCalculatedQuantities* ip_cq) override {
        cout << "Solver status: " << status << endl;
        cout << "Optimal u: " << x[0] << endl;
        cout << "Final cost: " << obj_value << endl;

        // Store solution
        solution_ = x[0];
    }
private:
    VectorXd B_;
    VectorXd c_;
    double solution_;
};

vector<float> plant(const vector<float>& x, const vector<float>& u) {
    /*
    Nonlinear kinematic equations
    \dot{\psi_1} = (v1 / L1) * tan \delta
    \dot{\psi_2} = (v1 / L2) * sin \theta - (v1*h / ((L1 * L2)) * tan \delta * cos \theta
    \dot{y_2} = (v1*cos \theta + ((v1*h*tan \delta) / (L1) * sin \theta) * sin \psi_2

    \theta = \psi_1 - \psi_2
    v2 = v1 * cos \theta + (v1*h*tan \delta / L1 ) * sin \theta
    */

    // Implement the above equations
    float v1 = -2.012; // m/s
    float h = -0.29; // m
    float L1 = 5.74; // m
    float L2 = 10.192; // m
    
    vector<float> xd;
    xd.push_back((v1 / L1) * tan(u[0]));

    float theta = x[0] - x[1];

    xd.push_back((v1 / L2) * sin(theta) - (v1 * h / (L1 * L2) * tan(u[0]) * cos(theta)));
    
    float v2 = v1 * cos(theta) + (v1 * h * tan(u[0]) / L1) * sin(theta);

    xd.push_back(v2 * sin(x[1]));
    return xd;
}

vector<vector<float>> get_reference_trajectory() {
        // Read .csv file 
    std::ifstream f{"trajectory_data.csv"};
    vector<vector<float>> x_r_t;
    string line;

    // Skip the header line
    std::getline(f, line);

    while (std::getline(f, line)) {
        vector<float> x_r;
        std::istringstream iss(line);
        string token;
        while (std::getline(iss, token, ',')) {
            try {
                x_r.push_back(std::stof(token));
            } catch (const std::invalid_argument& e) {
                cerr << "Invalid argument: " << token << " cannot be converted to float" << endl;
                // Handle the error as needed, e.g., skip the token or exit
                return {};
            } catch (const std::out_of_range& e) {
                cerr << "Out of range: " << token << " is out of range for float" << endl;
                // Handle the error as needed, e.g., skip the token or exit
                return {};
            }
        }
        x_r_t.push_back(x_r);
    }
    return x_r_t;
}

vector<float> controller(vector<float> x, vector<float> x_r) {
    vector<float> K = {-27.60653245, 99.8307537, -7.85407596}; 
    vector<float> u_fb(1, 0.0);
    vector<float> u_ff(1, 0.0);
    vector<float> u(1, 0.0);

    for (int i=0; i<3; ++i) {
        u_fb[0] -= K[i] * (x[i] - x_r[i]);
    }

    /*
    Bu_ff = -Axr + dxr = 0

    A =  [[ 0.          0.          0.        ]
         [-0.19740973  0.19740973  0.        ]
         [ 0.         -2.012       0.        ]]

    B = [[-0.35052265]
        [-0.00997366]
        [ 0.        ]]

    cost = | B*u - c| ^2
    cost = (B*u-c)^T * (B*u-c)
         = u^T * B^T * B * u - 2u^T*B^T*c + C^T * c
         = sum{ (B(i)*u-c(i))^2}

    derivative = 2*(B^T*Bu)-2*B^T*c
    u = (B^T*B)^-1 * B^T*c 
    */

    // Minimize cost function cost = | B*u - c| ^2 using Ipopt
    MatrixXd A = MatrixXd::Constant(3, 3, 0.0);
    A(1, 0) = -0.19740973;
    A(1, 1) = 0.19740973;
    A(2, 1) = -2.0120;

    VectorXd B(3);
    B << -0.35052265, -0.00997366, 0;

    VectorXd x_rr = VectorXd(3);
    x_rr << x_r[0], x_r[1], x_r[2];

    VectorXd xd_rr = VectorXd(3);
    xd_rr << x_r[3], x_r[4], x_r[5];

    VectorXd c = A * x_rr - xd_rr;

    SmartPtr<TNLP> mynlp = new QuadraticNLP(B, c);

    SmartPtr<IpoptApplication> app = IpoptApplicationFactory();
    app->Initialize();

    // Solve
    ApplicationReturnStatus status = app->OptimizeTNLP(mynlp);
    if (status == Solve_Succeeded) {
        cout << "Solve succeeded!" << endl;
    }
    else {
        cout << "Solve failed with status: " << status << endl;
    }

    u_ff[0] = dynamic_cast<QuadraticNLP*>(GetRawPtr(mynlp))->getSolution();

    u[0] = u_fb[0] + u_ff[0];

    return u;
}


int main() {
    // Read vars from config file
    /*
    std::ifstream f{"config.json"};
    json config = json::parse(f);
    */

    float T = 40.0; // seconds for total time of motion profile
    float dt = 0.08; // seconds
    float S1 = 0.2; // psi_1 [radians]
    float S2 = 0.2; // psi_2 [radians]
    float S3 = 5.0; // y2 [m]
    /*
    vector<float> x = {1*S1, 1*S2, -1 * S3}; // Initial state
    
    vector<vector<float>> trajectory = reference_trajectory_generation(dt, T, S1, S2, S3);
    */
    vector<vector<float>> trajectory = get_reference_trajectory();
    vector<float> x = {trajectory[0][0], trajectory[0][1], trajectory[0][2]}; // Initial state

    vector<float> u, xd;
    vector<vector<float>> result;
    result.push_back({x[0], x[1], x[2], 0});

    for (const auto& x_r: trajectory) {
        u = controller(x, x_r);
        xd = plant(x, u);

        // Euler integration
        for (int i = 0; i < 3; ++i) {
            x[i] += xd[i] * dt;
        }
        result.push_back({x[0], x[1], x[2], u[0]});
    }

    // Export reference trajectory
    std::ofstream out_path{"ref_trajectory.csv"};
    out_path << "psi_1" << "," << "psi_2" << "," << "y2" << '\n';
    // Iterate through the trajectory, ensuring no newline at the end of the file  
    for (size_t i = 0; i < trajectory.size(); ++i) {  
        const auto& x_r = trajectory[i];  
        out_path << x_r[0] << "," << x_r[1] << "," << x_r[2];  
        if (i != trajectory.size() - 1) {  
            out_path << '\n';  
        }  
    }  
    
    // Export x and u
    std::ofstream out_motion{"motion.csv"};
    out_motion << "x0" << "," << "x1" << "," << "x2" << "," << "u" << '\n';
    for (const auto& x: result) {
        if (x == result.back()) {
            out_motion << x[0] << "," << x[1] << "," << x[2] << "," << x[3];
        }
        else {
            out_motion << x[0] << "," << x[1] << "," << x[2] << "," << x[3] << '\n';
        }
    }
    return 0;
}