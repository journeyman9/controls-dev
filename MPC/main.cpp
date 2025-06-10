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
    QuadraticNLP(const MatrixXd& Abar, const MatrixXd& Bbar, const VectorXd& xo, const MatrixXd Qbar, const MatrixXd Rbar, const int N): Abar_(Abar), Bbar_(Bbar), xo_(xo), Qbar_(Qbar), Rbar_(Rbar), N_(N), solution_(0.0) {}

    // Public getter
    double getSolution() const {
        return solution_;
    }

    bool get_nlp_info(Index& n, Index& m, Index& nnz_jac_g, Index& nnz_h_lag, IndexStyleEnum& index_style) override {
        /*
        This method is called by Ipopt to get the problem information.
        params:
        - n: number of variables
        - m: number of constraints
        - nnz_jac_g: number of non-zero entries in the Jacobian of constraints
        - nnz_h_lag: number of non-zero entries in the Hessian of the Lagrangian
        - index_style: style of indexing (C_STYLE or FORTRAN_STYLE)
        */

        n = N_;
        m = 0;
        nnz_jac_g = 0;
        nnz_h_lag = N_*(N_+1)/2; // Hessian is symmetric, so we only need to store the upper triangle
        index_style = C_STYLE;
        return true;
    }

    bool get_bounds_info(Index n, Number* x_l, Number* x_u, Index m, Number *g_l, Number* g_u) override {
        /*
        This method is called by Ipopt to get the bounds on variables and constraints.
        params:
        - n: number of variables
        - x_l: lower bounds on variables
        - x_u: upper bounds on variables
        - m: number of constraints
        - g_l: lower bounds on constraints
        - g_u: upper bounds on constraints
        */
        for (Index i = 0; i < n; ++i) {
            x_l[i] = -0.78539816; // Lower bound on variables
            x_u[i] = 0.78539816;  // Upper bound on variables
        }
        return true;
    }

    bool get_starting_point(Index n, bool init_x, Number* x, bool init_z, Number* z_L, Number* z_U, Index m, bool init_lambda, Number *lambda) override {
        /*
        This method is called by Ipopt to get the initial guess for the variables.
        params:
        - n: number of variables
        - init_x: whether to initialize x
        - x: initial guess for variables
        - init_z: whether to initialize z_L and z_U
        - z_L: lower bounds on multipliers
        - z_U: upper bounds on multipliers
        - m: number of constraints
        - init_lambda: whether to initialize lambda
        - lambda: initial guess for multipliers on constraints
        */
        for (Index i = 0; i < n; ++i) {
            if (init_x) {
                x[i] = 0.0; // Initial guess for ubar
            }
            if (init_z) {
                z_L[i] = 0.0; // Lower bounds on multipliers
                z_U[i] = 0.0; // Upper bounds on multipliers
            }
        }
        return true;
    }

    bool eval_f(Index n, const Number* x, bool new_x, Number& obj_value) override {
        /*
        This method is called by Ipopt to evaluate the objective function.
        params:
        - n: number of variables
        - x: current values of the variables
        - new_x: whether x is new
        - obj_value: value of the objective function 
        */
        // Cost function: cost = ubar^T(Bbar^TQbarBbar + Rbar)ubar + 2 ubar^T Bbar^TQbarAbarxo + xo^T Abar^T Qbar Abar xo
        
        // I think just bad naming of vars
        VectorXd ubar_ = VectorXd(N_);
        for (int i = 0; i < N_; ++i) {
            ubar_[i] = x[i];
        }

        obj_value = ubar_.transpose()*(Bbar_.transpose() * Qbar_ * Bbar_ + Rbar_) * ubar_;
        obj_value += 2.0 * ubar_.transpose() * Bbar_.transpose() * Qbar_ * Abar_ * xo_;
        obj_value += xo_.transpose() * Abar_.transpose() * Qbar_ * Abar_ * xo_;
        return true;
    }

    bool eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f) override {
        /*
        This method is called by Ipopt to evaluate the gradient of the objective function.
        params:
        - n: number of variables
        - x: current values of the variables
        - new_x: whether x is new
        - grad_f: gradient of the objective function 
        */
        // Gradient of the cost function with respect to ubar: cost = = 2(Bbar^T Qbar Bbar + Rbar)ubar + 2 Bbar^T Qbar Abar xo 
        
        VectorXd ubar_ = VectorXd(N_);
        for (int i = 0; i < N_; ++i) {
            ubar_[i] = x[i];
        }

        VectorXd grad = 2.0 * (Bbar_.transpose() * Qbar_ * Bbar_ + Rbar_) * ubar_ + 2.0 * Bbar_.transpose() * Qbar_ * Abar_ * xo_;

        for (int i = 0; i < N_; ++i) {
            grad_f[i] = grad[i];
        }
        return true;
    }

    bool eval_g(Index n, const Number* x, bool new_x, Index m, Number* g) override {
        /*
        This method is called by Ipopt to evaluate the constraints.
        params:
        - n: number of variables
        - x: current values of the variables
        - new_x: whether x is new
        - m: number of constraints
        - g: values of the constraints
        */
        
        /*
        MatrixXd G = MatrixXd::Zero(m, n);
        G[0, 0] = 1.0;
        G[0, 1] = -1.0;
        G[1, 0] = -1.0;
        G[1, 1] = 1.0;
        
        VectorXd ubar_ = VectorXd(N_);
        for (int i = 0; i < N_; ++i) {
            ubar_[i] = x[i];
        }

        VectorXd h = VectorXd::Zero(m);
        for (int i = 0; i < m; ++i) {
            h[i] = 3.1415926 / 2.0;
        }

        auto constraints = G * Bbar_ * ubar_ - h - G * Abar_ * xo_;
        
        g[0] = constraints[0];
        g[1] = constraints[1];
        */

        return true;
    }

    bool eval_jac_g(Index n, const Number* x, bool new_x, Index m, Index nnz_jac_g, Index* iRow, Index* jCol, Number* values) override {
        /*
        This method is called by Ipopt to evaluate the Jacobian of the constraints.
        params:
        - n: number of variables
        - x: current values of the variables
        - new_x: whether x is new
        - m: number of constraints
        - nnz_jac_g: number of non-zero entries in the Jacobian
        - iRow: row indices of the non-zero entries
        - jCol: column indices of the non-zero entries
        - values: values of the non-zero entries
        */
        
        /*
        MatrixXd G = MatrixXd::Zero(m, n);
        G[0, 0] = 1.0;
        G[0, 1] = -1.0;
        G[1, 0] = -1.0;
        G[1, 1] = 1.0;

        auto Jacobian = G*Bbar_;

        if (values==nullptr) {
            // Return structure of the Jacobian (lower triangular)
            Index idx = 0;
            for (Index col = 0; col < n; col++) {
                for (Index row = 0; row < n; row++) {  // row starts from 0
                    iRow[idx] = row;
                    jCol[idx] = col;
                    idx++;
                }
            }
        } 
        else {
            // Return values of the Jacobian (lower triangular)
            for (Index col = 0; col < n; col++) {
                for (Index row = 0; row < n; row++) {  // row starts from 0
                    if (row < col) {
                        values[row * n + col] = 0.0; // Lower triangular part is zero
                    } else {
                        values[row * n + col] = Jacobian(row, col);
                    }
                }
            }
        }
        */
        return true;
    }

    bool eval_h(Index n, const Number* x, bool new_x, Number obj_factor, Index m, const Number* lambda, bool new_lambda, Index nnz_h_lag, Index* iRow, Index* jCol, Number* values) override {
        /*
        This method is called by Ipopt to evaluate the Hessian of the Lagrangian.
        params:
        - n: number of variables
        - x: current values of the variables
        - new_x: whether x is new
        - obj_factor: factor for the objective function
        - m: number of constraints
        - lambda: multipliers on the constraints
        - new_lambda: whether lambda is new
        - nnz_h_lag: number of non-zero entries in the Hessian
        - iRow: row indices of the non-zero entries
        - jCol: column indices of the non-zero entries
        - values: values of the non-zero entries
        */
        // Hessian of the cost wrt ubar: // H = 2(Bbar^T Qbar Bbar + Rbar)
        if (values == nullptr) {
            // Return structure of the Hessian (lower triangular)
            Index idx = 0;
            for (Index col = 0; col < n; col++) {
                for (Index row = col; row < n; row++) {  // row starts from col
                    iRow[idx] = row;
                    jCol[idx] = col;
                    idx++;
                }
            }
        } 
        else {
            // Return values of the Hessian (lower triangular)
            MatrixXd H = 2.0 * (Bbar_.transpose() * Qbar_ * Bbar_ + Rbar_);
            Index idx = 0;
            for (Index col = 0; col < n; col++) {
                for (Index row = col; row < n; row++) {  // row starts from col
                    values[idx] = obj_factor * H(row, col);
                    idx++;
                }
            }
        }

        // No contribution from constraints because Hessian of constraint is 0 

        return true;
    }

    void finalize_solution(SolverReturn status, Index n, const Number* x, const Number* z_L, const Number* z_U, Index m, const Number* g, const Number* lambda, Number obj_value, const IpoptData* ip_data, IpoptCalculatedQuantities* ip_cq) override {
        /*
        This method is called by Ipopt to finalize the solution.
        params:
        - status: status of the solver
        - n: number of variables
        - x: optimal values of the variables
        - z_L: lower bounds on multipliers
        - z_U: upper bounds on multipliers
        - m: number of constraints
        - g: values of the constraints
        - lambda: multipliers on the constraints
        - obj_value: value of the objective function at the solution
        - ip_data: IPOPT data
        - ip_cq: IPOPT calculated quantities
        */
        //cout << "Solver status: " << status << endl;
        //cout << "Optimal u: " << x[0] << endl;
        //cout << "Final cost: " << obj_value << endl;
        for (Index i = 0; i < n; ++i) {
            cout << "u[" << i << "] = " << x[i] << endl;
        }

        // Store solution
        solution_ = x[0];
    }
private:
    MatrixXd Abar_;
    MatrixXd Bbar_;
    VectorXd xo_;
    MatrixXd Qbar_;
    MatrixXd Rbar_;
    int N_;
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

vector<float> controller(vector<float> x, vector<float> x_r, int N=1) {
    vector<float> u(1, 0.0);

    // Minimize cost = ubar^T(Bbar^TQbarBbar + Rbar)ubar + 2 ubar^TBbar^TQbarAbarxo + xo^T Abar^T Qbar Abar xo
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

    VectorXd x_ = VectorXd(3);
    x_ << x[0], x[1], x[2];

    VectorXd xo = x_ - x_rr;

    // Cost matrices
    MatrixXd Q = MatrixXd::Identity(3, 3);
    MatrixXd R = MatrixXd::Identity(1, 1);

    MatrixXd Qbar = MatrixXd::Zero(3*N, 3*N);
    for(int i = 0; i < N; i++) {
        Qbar.block<3,3>(i*3, i*3) = Q;
    }

    // Similarly for Rbar
    MatrixXd Rbar = MatrixXd::Zero(N, N);
    for(int i = 0; i < N; i++) {
        Rbar(i,i) = R(0,0);
    }

    // Construct Abar and Bbar
    MatrixXd Abar = MatrixXd::Zero(3*N, 3);
    MatrixXd Bbar = MatrixXd::Zero(3*N, N);
    MatrixXd A_power = MatrixXd::Identity(3, 3);
    A_power = A;
    for(int i = 0; i < N; i++) {
        Abar.block<3, 3>(i*3, 0) = A_power;
        A_power *= A;
    }

    /*
    Bbar = [
        B, 0, 0, ..., 0; 
        AB, B , 0, ..., 0;
        A^2*B, A*B, B, ..., 0;
        A^(N-1)*B, A^(N-2)*B, A^(N-3)*B, ..., B
    ]
    */
    A_power = A;
    for(int i = 0; i < N; i++) {
        for(int j = 0; j <= i; j++) {
            Bbar.block<3, 1>(i*3, j) = A_power * B;
        }
        A_power = A_power * A; // Update A_power to A^(j+1)
    }
    /*
    std::cout << "Abar dimensions: " << Abar.rows() << "x" << Abar.cols() << std::endl;
    std::cout << "Bbar dimensions: " << Bbar.rows() << "x" << Bbar.cols() << std::endl;
    std::cout << "Qbar dimensions: " << Qbar.rows() << "x" << Qbar.cols() << std::endl;
    std::cout << "Rbar dimensions: " << Rbar.rows() << "x" << Rbar.cols() << std::endl;
    */

    /*
    std::cout << "Abar:\n" << Abar << std::endl;
    std::cout << "Bbar:\n" << Bbar << std::endl;
    
    MatrixXd H = 2.0 * (Bbar.transpose() * Qbar * Bbar + Rbar);
    std::cout << "Hesssian:\n" << H << std::endl;
    */

    // Break code exit
    //assert(false);

    SmartPtr<TNLP> mynlp = new QuadraticNLP(Abar, Bbar, xo, Qbar, Rbar, N);

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

    u[0] = dynamic_cast<QuadraticNLP*>(GetRawPtr(mynlp))->getSolution();
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
    int N = 6; // Horizon length
    /*
    vector<float> x = {1*S1, 1*S2, -1 * S3}; // Initial state
    
    vector<vector<float>> trajectory = reference_trajectory_generation(dt, T, S1, S2, S3);
    */
    vector<vector<float>> trajectory = get_reference_trajectory();
    vector<float> x = {trajectory[0][0], trajectory[0][1], trajectory[0][2]}; // Initial state

    vector<float> u, xd;
    vector<vector<float>> result;
    result.push_back({x[0], x[1], x[2], 0});

    std::chrono::duration<float> duration;
    vector<std::chrono::duration<float>> all_duration;
    for (const auto& x_r: trajectory) {
        //u = controller(x, x_r);

        auto t0 = std::chrono::high_resolution_clock::now();
        u = controller(x, x_r, N);
        auto t1 = std::chrono::high_resolution_clock::now();

        duration = t1 - t0;
        all_duration.push_back(duration);

        xd = plant(x, u);

        // Euler integration
        for (int i = 0; i < 3; ++i) {
            x[i] += xd[i] * dt;
        }
        result.push_back({x[0], x[1], x[2], u[0]});
    }

    std::chrono::duration<float> total_time(0);
    total_time = std::accumulate(all_duration.begin(), all_duration.end(), total_time);
    cout << "Total time: " << total_time.count() << " seconds" << endl;

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