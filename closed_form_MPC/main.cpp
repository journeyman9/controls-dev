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

vector<vector<float>> get_reference_trajectory(string filename) {
    // Read .csv file
    std::ifstream f{filename};
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

vector<float> controller(vector<float> x, vector<float> x_r, int N, MatrixXd A, VectorXd B, MatrixXd Q, MatrixXd R) {
    vector<float> u(1, 0.0);
    VectorXd ubar = VectorXd::Zero(N); // Control input vector

    // Minimize cost = ubar^T(Bbar^TQbarBbar + Rbar)ubar + 2 ubar^TBbar^TQbarAbarxo + xo^T Abar^T Qbar Abar xo

    VectorXd x_rr = VectorXd(3);
    x_rr << x_r[0], x_r[1], x_r[2];

    VectorXd xd_rr = VectorXd(3);
    xd_rr << x_r[3], x_r[4], x_r[5];

    VectorXd x_ = VectorXd(3);
    x_ << x[0], x[1], x[2];

    VectorXd xo = x_ - x_rr;
    
    // --- terminal weight --------------------------------------------------
    // P is the infinite–horizon LQR cost matrix: AᵀPA − AᵀPB(R+ BᵀPB)⁻¹BᵀPA + Q = P
    MatrixXd P = Q;                       // initialise
    for (int k = 0; k < 200; ++k) {       // small fixed-point iteration
        MatrixXd Ktmp = (R + B.transpose()*P*B).inverse()*B.transpose()*P*A;
        MatrixXd Pnext = Q + A.transpose()*(P - P*B*Ktmp)*A;
        if ((Pnext-P).norm() < 1e-9) {
            P = Pnext;
            break;
        }
        P = Pnext;
    }

    // fill Q̄ and overwrite the last 3×3 block with P
    MatrixXd Qbar = MatrixXd::Zero(3*N, 3*N);
    for (int i = 0; i < N-1; ++i)
        Qbar.block<3,3>(3*i, 3*i) = Q;
    Qbar.block<3,3>(3*(N-1), 3*(N-1)) = P;   // terminal cost
    
    /*
    MatrixXd Qbar = MatrixXd::Zero(3*N, 3*N);
    for(int i = 0; i < N; i++) {
        Qbar.block<3,3>(i*3, i*3) = Q;
    }
    */

    // Similarly for Rbar
    MatrixXd Rbar = MatrixXd::Zero(N, N);
    for(int i = 0; i < N; i++) {
        Rbar(i,i) = R(0,0);
    }

    // Construct Abar and Bbar
    MatrixXd Abar = MatrixXd::Zero(3*N, 3);
    MatrixXd Bbar = MatrixXd::Zero(3*N, N);
    MatrixXd A_power = A;
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
    for (int i = 0; i < N; i++) {
        for (int j = 0; j <= i; j++) {
            // Calculate A^(i-j) * B
            MatrixXd A_power = MatrixXd::Identity(3, 3);
            for (int k = 0; k < (i - j); k++) {
                A_power *= A;
            }
            Bbar.block<3, 1>(i*3, j) = A_power * B;
        }
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
    
    // ubar = -(Bbar^T * Qbar * Bbar + Rbar) ^-1 * Bbar^T * Qbar * Abar * xo
    MatrixXd K = (Bbar.transpose() * Qbar * Bbar + Rbar).inverse() * Bbar.transpose() * Qbar * Abar;
    ubar = -1.0 * K * xo;

    cout << "K_row0: " << K.row(0) << endl;
    cout << "ubar: " << ubar.head(2).transpose() << ".." << ubar.tail(2).transpose() << endl;
    
    // Check if the closed-loop system is stable using first row of K
    // The closed-loop system is stable if the eigenvalues of A - B*K are all less than 1 
    // This is a simplified check, assuming the first row of K is used for control
    MatrixXd check = A - (B * K.row(0));
    
    /*
    cout << "Check A-BK:" << "\n" << check << endl;

    Eigen::EigenSolver<Eigen::Matrix3d> solver(check);
    if (solver.info() == Eigen::Success) {
        std::cout << "Eigenvalues:\n" << solver.eigenvalues() << std::endl;
    } else {
        std::cerr << "Failed to compute eigenvalues." << std::endl;
    }
    */

    // Assert eigenvalues are all less than 1
    Eigen::EigenSolver<Eigen::Matrix3d> solver(check);
    for (int i = 0; i < solver.eigenvalues().size(); ++i) {
        if (std::abs(solver.eigenvalues()[i].real()) >= 1.0) {
            std::cerr << "Closed-loop system is unstable!" << std::endl;
            assert(false);
        }
    }

    u[0] = std::clamp(ubar[0],-0.78539816, 0.78539816);

    return u;
}


int main() {
    // Read vars from config file
    std::ifstream f{"config.json"};
    json config = json::parse(f);

    float T = 40.0; // seconds for total time of motion profile
    float dt = config["dt"]; // seconds
    
    int N = config["N"]; // Horizon length
    
    MatrixXd A = MatrixXd::Identity(3, 3);
    A(0, 0) = 1.0;
    A(1, 0) = -0.01591814;
    A(1, 1) = 1.01591814;
    A(2, 0) = 0.00127772;
    A(2, 1) = -0.16223772;
    A(2, 2) = 1.0;

    VectorXd B(3);
    B << -0.02804181, -0.00058163, 0.00005263;
    
    MatrixXd Q = MatrixXd::Identity(3, 3);
    MatrixXd R = MatrixXd::Identity(1, 1);

    Q(0, 0) = static_cast<float>(config["Q"][0][0]) * dt;
    Q(1, 1) = static_cast<float>(config["Q"][1][1]) * dt;
    Q(2, 2) = static_cast<float>(config["Q"][2][2]) * dt;

    R(0, 0) = static_cast<float>(config["R"][0][0]) * dt;

    vector<vector<float>> trajectory = get_reference_trajectory(config["reference_trajectory_file"]);
    vector<float> x = {trajectory[0][0], trajectory[0][1], trajectory[0][2]}; // Initial state

    vector<float> u, xd;
    vector<vector<float>> result;
    result.push_back({x[0], x[1], x[2], 0});

    std::chrono::duration<float> duration;
    vector<std::chrono::duration<float>> all_duration;
    for (const auto& x_r: trajectory) {
        //u = controller(x, x_r);

        auto t0 = std::chrono::high_resolution_clock::now();
        u = controller(x, x_r, N, A, B, Q, R);
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