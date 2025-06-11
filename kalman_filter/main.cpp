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
#include <random>
 
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

vector<vector<float>> get_reference_trajectory(string filename = "trajectory_data.csv") {
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

vector<float> controller(vector<float> x, vector<float> x_r) {
    vector<float> K = {-27.60653245, 99.8307537, -7.85407596}; 
    vector<float> u(1, 0.0);

    for (int i=0; i<3; ++i) {
        u[0] -= K[i] * (x[i] - x_r[i]);
    }

    return u;
}

void kalman_filter(VectorXd& x_hat, const VectorXd& u_hat, MatrixXd& P, const VectorXd& z, const MatrixXd& Q, const MatrixXd& R, const MatrixXd& H, const MatrixXd& F, const VectorXd& G, const VectorXd& wn, const vector<float> x_r) {
    VectorXd x_rr = VectorXd(3);
    x_rr << x_r[0], x_r[1], x_r[2];

    VectorXd xd_rr = VectorXd(3);
    xd_rr << x_r[3], x_r[4], x_r[5];

    // TODO: Update step need to happen over multiple measurements? Warm start?
    // Update step
    VectorXd y = z - H * x_hat; // Measurement residual
    MatrixXd S = H * P * H.transpose() + R; // Residual covariance
    MatrixXd Kn = P * H.transpose() * S.inverse(); // Kalman gain

    x_hat = x_hat + Kn * y; // Updated state estimate
    P = (MatrixXd::Identity(3, 3) - (Kn * H)) * P * (MatrixXd::Identity(3, 3) - (Kn * H)).transpose() + Kn * R * Kn.transpose();
    
    // Prediction step
    x_hat = F*x_hat + G*u_hat;
    P = F*P*F.transpose() + Q;
}


int main() {
    // Read vars from config file
    /*
    std::ifstream f{"config.json"};
    json config = json::parse(f);
    */

    float T = 40.0; // seconds for total time of motion profile
    float dt = 0.08; // seconds
    vector<vector<float>> trajectory = get_reference_trajectory("trajectory_data.csv");
    vector<float> x = {trajectory[0][0], trajectory[0][1], trajectory[0][2]}; // Initial state

    vector<float> u{0};
    vector<float> xd;
    vector<vector<float>> result;
    result.push_back({x[0], x[1], x[2], 0});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0, 0.04);

    //Setup Kalman filter
    MatrixXd F = MatrixXd::Identity(3, 3);
    F(1, 0) = 1.0;
    F(1, 1) = 1.01591814;
    F(2, 1) = 1.0;
    F(1, 0) = -0.01591814;
    F(2, 0) = 0.00127772;
    F(2, 1) = -0.16223772;

    VectorXd G(3);
    G << -0.02804181, -0.00058163, 0.00005263;
    
    //F = F * dt + MatrixXd::Identity(3, 3); // Discrete state transition matrix manually
    //G *= dt; // Discrete state input matrix manually from continuous

    VectorXd wn = VectorXd::Zero(3); // Process noise, assuming no process noise for simplicity
    VectorXd z(3); // Measurement vector
    z << x[0], x[1], x[2]; // Measurement vector
    z[1] += std::clamp(dist(gen), -0.17f, 0.17f);
    z[2] += std::clamp(dist(gen), -0.3f, 0.3f);
    
    VectorXd x_hat(3); // Initial state estimate

    MatrixXd Q(3, 3); // Process noise covariance
    // TODO: how calculate Q?
    Q << 0.01, 0.0, 0.0,
         0.0, 0.01, 0.0,
         0.0, 0.0, 0.01; // Process noise covariance matrix
    
    MatrixXd R(3, 3); // Measurement noise covariance
    // TODO: how calculate R?
    R << pow(0.04, 2), 0.0, 0.0,
         0.0, pow(0.04, 2), 0.0,
         0.0, 0.0, pow(0.04, 2); // Measurement noise covariance matrix

    MatrixXd H(3, 3); // Measurement matrix
    H << 1.0, 0.0, 0.0,
         0.0, 1.0, 0.0,
         0.0, 0.0, 1.0; // Assuming direct measurement of state
    
    MatrixXd P(3, 3);
    P << 1.0, 0.0, 0.0,
         0.0, 1.0, 0.0,
         0.0, 0.0, 1.0; // Initial covariance matrix
    
    VectorXd u_hat(1);
    u_hat << 0.0;
    
    /*
    // TODO: Does calibration make sense, measurement_data needs u to pass in u_hat
    x_hat << z[0], z[1], z[2]; // Initial state estimate from measurement
    vector<vector<float>> calibration = get_reference_trajectory("measurement_data.csv");
    for (auto& x_m: calibration) {
        // Add noise to the state
        z << x_m[0], x_m[1], x_m[2]; // Measurement vector
        z[1] += std::clamp(dist(gen), -0.17f, 0.17f);
        z[2] += std::clamp(dist(gen), -0.3f, 0.3f);

        u_hat << x_m[6]; // Convert u to VectorXd for kalman_filter
        // Kalman filter step
        kalman_filter(x_hat, u_hat, P, z, Q, R, H, F, G, wn, x_m);
        //x_hat passed by reference
        
        x_hat = F*x_hat + G*x_m[6];
        P = F*P*F.transpose() + Q;
    }
    */

    // If only using one sample
    x_hat = F*z + G*u_hat;
    P = F*P*F.transpose() + Q;

    for (const auto& x_r: trajectory) {
        // Add noise to the state
        z << x[0], x[1], x[2]; // Measurement vector
        //z[0] += std::clamp(dist(gen), -0.17f, 0.17f);
        z[1] += std::clamp(dist(gen), -0.17f, 0.17f);
        z[2] += std::clamp(dist(gen), -0.3f, 0.3f);

        // Kalman filter step
        u_hat << u[0]; // Convert u to VectorXd for kalman_filter
        kalman_filter(x_hat, u_hat, P, z, Q, R, H, F, G, wn, x_r);
        //x_hat passed by reference
        std::vector<float> x_hat_std(x_hat.data(), x_hat.data() + x_hat.size());
        u = controller(x_hat_std, x_r);

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
    cout << "Motion profile generated successfully!" << endl;
    return 0;
}