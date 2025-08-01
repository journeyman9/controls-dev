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

#define _USE_MATH_DEFINES
#include <cmath>
 
using Eigen::MatrixXd;
using Eigen::VectorXd;

using namespace std;
using json = nlohmann::json;

// wrapped angle difference  (−π , π]
inline float safe_minus(float r, float c)
{
    const float d = r - c;
    if (std::fabs(d) <= static_cast<float>(M_PI)) return d;
    return d - 2.0f * static_cast<float>(M_PI) * std::copysign(1.0f, d);
}

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

    // Discrete
    /*
    MatrixXd A = MatrixXd::Identity(3, 3);
    VectorXd B = VectorXd(3);
    // Implement state space discrete model
    A(0, 0) = 1.0;
    A(1, 0) = -0.00158053;
    A(1, 1) = 1.00158053;
    A(2, 0) = 0.00001272;
    A(2, 1) = -0.01610872;
    A(2, 2) = 1.0;
    
    B << -0.00280418, -0.00007764, 0.00000063;
    */
    
    // Continuous State space
    /*
    MatrixXd A = MatrixXd::Constant(3, 3, 0.0);
    A(1, 0) = -0.19740973;
    A(1, 1) = 0.19740973;
    A(2, 1) = -2.0120;

    VectorXd B(3);
    B << -0.35052265, -0.00997366, 0;
    */
    
    /*
    VectorXd x_vec(3);
    x_vec << x[0], x[1], x[2];
    VectorXd u_vec(1);
    u_vec << u[0];
    VectorXd xd_vec = A * x_vec + B * u_vec;
    vector<float> xd(3);
    xd[0] = xd_vec(0);
    xd[1] = xd_vec(1);
    xd[2] = xd_vec(2);
    */
    
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
    // Discrete .08
    //vector<float> K = {-20.26030997, 72.07448439, -5.57790699};
    // Discrete .008
    //vector<float> K = {-26.74194825, 96.56371101, -7.5860704};
    vector<float> u(1, 0.0);

    // Use safe_minus to handle angle wrapping
    x[0] = safe_minus(x[0], x_r[0]);
    x[1] = safe_minus(x[1], x_r[1]);

    // Calculate control input where x[0] is psi_1, x[1] is psi_2, x[2] is y_2
    u[0] = -K[0] * x[0] - K[1] * x[1] - K[2] * (x[2] - x_r[2]);
    //for (int i=0; i<3; ++i) {
    //    u[0] -= K[i] * (x[i] - x_r[i]);
    //}

    return u;
}

MatrixXd kalman_filter(VectorXd& x_hat, const VectorXd& u_k, MatrixXd& P, const VectorXd& z, const MatrixXd& Q, const MatrixXd& R, const MatrixXd& H, const MatrixXd& F, const VectorXd& G) {
    // Prediction step
    x_hat = F*x_hat + G*u_k;
    P = F*P*F.transpose() + Q;

    VectorXd y = z - (H*x_hat); // Innovation
    y[0] = safe_minus(z[0], (H*x_hat)[0]); // Wrap angle difference to (-π, π]
    y[1] = safe_minus(z[1], (H*x_hat)[1]); // Wrap angle difference to (-π, π]
    // Update step
    MatrixXd Kf = P * H.transpose() * (H * P * H.transpose() + R).inverse(); // Kalman gain
    x_hat = x_hat + Kf * (y); // Output state
    P = (MatrixXd::Identity(3, 3) - (Kf * H)) * P * (MatrixXd::Identity(3, 3) - (Kf * H)).transpose() + Kf * R * Kf.transpose();
    return Kf;
}

Eigen::MatrixXd get_covariance_matrix(const Eigen::MatrixXd& mat) {
    Eigen::RowVectorXd mean = mat.colwise().mean();
    Eigen::MatrixXd centered = mat.rowwise() - mean;
    Eigen::MatrixXd covariance = (centered.adjoint() * centered) / double(mat.rows() - 1);
    return covariance;
}

auto wrap = [](float a){ return std::atan2(std::sin(a), std::cos(a)); };

int main() {
    // Read vars from config file
    /*
    std::ifstream f{"config.json"};
    json config = json::parse(f);
    */

    float dt = 0.008; // seconds
    float dt_controller = 0.08;
    int steps_per_control = int(dt_controller / dt);
    vector<vector<float>> trajectory = get_reference_trajectory("trajectory_data.csv");
    vector<float> x = {trajectory[0][0], trajectory[0][1], trajectory[0][2]}; // Initial state
    
    //cout << "Initial state: " << x[0] << ", " << x[1] << ", " << x[2] << endl;
    //assert(false);
    // TODO: remove
    //x[2] += 0.1;

    vector<float> u{0};
    vector<float> xd;
    vector<vector<float>> result;
    vector<vector<float>> kalman_result;
    result.push_back({x[0], x[1], x[2], 0});

    std::random_device rd;
    //std::mt19937 gen(rd());
    std::mt19937 gen(42);
    //std::normal_distribution<float> dist(0.0, 0.04);
    std::normal_distribution<float> yaw_noise(0.0, 0.02);
    std::normal_distribution<float> pos_noise(0.0, 0.25);
    //Uniform distribution for noise
    //std::uniform_real_distribution<float> yaw_noise(-0.05, 0.05);
    //std::uniform_real_distribution<float> pos_noise(-0.5, 0.5);
    
    std::normal_distribution<float> process_noise_psi(0.0, 0.1);   // Process noise for angles
    std::normal_distribution<float> process_noise_y(0.0, 0.3);     // Process noise for position

    //Setup Kalman filter
    MatrixXd F = MatrixXd::Identity(3, 3);
    // dt = 0.08
    /*
    F(0, 0) = 1.0;
    F(1, 0) = -0.01591814;
    F(1, 1) = 1.01591814;
    F(2, 0) = 0.00127772;
    F(2, 1) = -0.16223772;
    F(2, 2) = 1.0;
    */
    
    // dt = 0.008
    F(0, 0) = 1.0;
    F(1, 0) = -0.00158053;
    F(1, 1) = 1.00158053;
    F(2, 0) = 0.00001272;
    F(2, 1) = -0.01610872;
    F(2, 2) = 1.0;

    VectorXd G(3);
    // dt = 0.08
    //G << -0.02804181, -0.00058163, 0.00005263;
    
    // dt = 0.008 
    G << -0.00280418, -0.00007764, 0.00000063;
    
    //F = F * dt + MatrixXd::Identity(3, 3); // Discrete state transition matrix manually
    //G *= dt; // Discrete state input matrix manually from continuous

    VectorXd z(3); // Measurement vector
    z << x[0], x[1], x[2]; // Measurement vector
    //z[0] += std::clamp(dist(gen), -0.17f, 0.17f);
    //z[1] += std::clamp(dist(gen), -0.1f, 0.1f);
    //z[2] += std::clamp(dist(gen), -0.05f, 0.05f);
    //z[0] += yaw_noise(gen); // Add noise to psi_1
    //z[1] += yaw_noise(gen);
    //z[2] += pos_noise(gen);

    VectorXd x_hat(3); // Initial state estimate
    x_hat << z[0], z[1], z[2]; // Initial state estimate from measurement
    
    MatrixXd Q(3, 3); // Process noise covariance
    MatrixXd R(3, 3); // Measurement noise covariance
    
    // Calculate covariance matrices 
    /*
    vector<vector<float>> calibration = get_reference_trajectory("measurement_data.csv");
    Eigen::MatrixXd data(calibration.size(), 3);  
    Eigen::MatrixXd og_data(calibration.size(), 3);  
    for (size_t i = 0; i < calibration.size(); i++) {
        auto& x_m = calibration[i];
        
        // Populate original data
        og_data(i, 0) = x_m[0];
        og_data(i, 1) = x_m[1]; 
        og_data(i, 2) = x_m[2];
        
        // Add noise and populate noisy data
        //data(i, 0) = x_m[0];
        //data(i, 0) = x_m[0] + std::clamp(dist(gen), -0.1f, 0.1f);
        //data(i, 1) = x_m[1] + std::clamp(dist(gen), -0.1f, 0.1f);
        //data(i, 2) = x_m[2] + std::clamp(dist(gen), -0.05f, 0.05f);
        data(i, 0) = x_m[0] + yaw_noise(gen);
        data(i, 1) = x_m[1] + yaw_noise(gen);
        data(i, 2) = x_m[2] + pos_noise(gen);
    }
    
    // Calculate Process noise covariance Q
    Q = get_covariance_matrix(og_data);

    // Calculate Measurement noise covariance R
    R = get_covariance_matrix(data);
    */

    Q << 0.00001, 0.0, 0.0,
         0.0, 0.00001, 0.0,
         0.0, 0.0, 0.005; // Process noise covariance matrix
    
    R << pow(0.06, 2), 0.0, 0.0,
         0.0, pow(0.06, 2), 0.0,
         0.0, 0.0, pow(0.25, 2); // Measurement noise covariance matrix
    
    // Very small Q since there's no process noise in the plant model
    /*
    Q << 0.01, 0.0, 0.0,        // Very small psi_1 process noise
         0.0, 0.01, 0.0,        // Very small psi_2 process noise  
         0.0, 0.0, 0.1;        // Very small y2 process noise

    // Estimate R from actual noise samples
    int noise_samples = 1000;
    Eigen::MatrixXd noise_data(noise_samples, 3);
    
    for (int i = 0; i < noise_samples; i++) {
        noise_data(i, 0) = yaw_noise(gen);   // psi_1 noise
        noise_data(i, 1) = yaw_noise(gen);   // psi_2 noise
        noise_data(i, 2) = pos_noise(gen);   // y2 noise
    }
    
    R = get_covariance_matrix(noise_data);
    */

    MatrixXd H(3, 3); // Measurement matrix
    H << 1.0, 0.0, 0.0,
         0.0, 1.0, 0.0,
         0.0, 0.0, 1.0; // Assuming direct measurement of state
    
    MatrixXd P(3, 3);
    // Initial covariance matrix
    P << 0.001, 0.0, 0.0, // truck yaw
         0.0, 0.001, 0.0, // trailer yaw
         0.0, 0.0, 0.001;  // cross track error
    
    VectorXd u_k(1);
    
    /*
    float u_applied_calibration = 0.0f; // Initial control input
    // TODO: Does calibration make sense, measurement_data needs u to pass in u_k
    for (auto& x_m: calibration) {
        // Add noise to the state
        z << x_m[0], x_m[1], x_m[2]; // Measurement vector
        //z[0] += std::clamp(dist(gen), -0.1f, 0.1f);
        //z[1] += std::clamp(dist(gen), -0.1f, 0.1f);
        //z[2] += std::clamp(dist(gen), -0.05f, 0.05f);
        z[0] += yaw_noise(gen); // Add noise to psi_1
        z[1] += yaw_noise(gen);
        z[2] += pos_noise(gen);

        u_k << u_applied_calibration; // Convert u to VectorXd for kalman_filter
        // Kalman filter step
        MatrixXd Kf = kalman_filter(x_hat, u_k, P, z, Q, R, H, F, G);
        //x_hat passed by reference
        
        u_applied_calibration = x_m[6];
    }
    */

    // If only using one sample
    //x_hat = F*z + G*u_k;
    //P = F*P*F.transpose() + Q;

    /*
    vector<float> x_r = trajectory[0]; // Reference state for the first step
    MatrixXd Kf = kalman_filter(x_hat, u_k, P, z, Q, R, H, F, G);

    cout << "K: " << Kf << endl;
    
    MatrixXd check = F - (F * Kf * H);
    cout << "Check A-AKH:" << "\n" << check << endl;
    
    Eigen::EigenSolver<Eigen::MatrixXd> solver(check);
    if (solver.info() == Eigen::Success) {
        std::cout << "Eigenvalues:\n" << solver.eigenvalues() << std::endl;
    } else {
        std::cerr << "Failed to compute eigenvalues." << std::endl;
    }
    assert(false);
    */
    
    // TODO: does it make sense to precalculate Kf?
    //vector<float> x_r = trajectory[0]; // Reference state for the first step
    //MatrixXd Kf = kalman_filter(x_hat, u_k, P, z, Q, R, H, F, G);

    int count = 0;
    float u_applied = 0.0f;
    for (auto& x_r: trajectory) {
        xd = plant(x, u);

        // Euler integration
        for (int i = 0; i < 3; ++i) {
            x[i] += xd[i] * dt;
        }

        //Process noise
        /*
        x[0] += (xd[0] * dt) + process_noise_psi(gen) * dt;
        x[1] += (xd[1] * dt) + process_noise_psi(gen) * dt;
        x[2] += (xd[2] * dt) + process_noise_y(gen) * dt;
        */
        
        // Runge-Kutta 4th order (RK4) - more accurate
        /*
        vector<float> k1 = plant(x, u);
        vector<float> x_temp1(3);
        for(int i = 0; i < 3; ++i) x_temp1[i] = x[i] + 0.5 * dt * k1[i];
        
        vector<float> k2 = plant(x_temp1, u);
        vector<float> x_temp2(3);
        for(int i = 0; i < 3; ++i) x_temp2[i] = x[i] + 0.5 * dt * k2[i];
        
        vector<float> k3 = plant(x_temp2, u);
        vector<float> x_temp3(3);
        for(int i = 0; i < 3; ++i) x_temp3[i] = x[i] + dt * k3[i];
        
        vector<float> k4 = plant(x_temp3, u);
        
        for(int i = 0; i < 3; ++i) {
            x[i] += (dt/6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]);
        }
        */
        
        // For discrete sim
        /*
        x[0] = xd[0];
        x[1] = xd[1];
        x[2] = xd[2];
        */

        // Add noise to the state
        z << x[0], x[1], x[2]; // Measurement vector
        //z[0] += std::clamp(dist(gen), -0.1f, 0.1f);
        //z[1] += std::clamp(dist(gen), -0.1f, 0.1f);
        //z[2] += std::clamp(dist(gen), -0.05f, 0.05f);
        //z[0] += yaw_noise(gen);
        //z[1] += yaw_noise(gen);

        z[0] = wrap(x[0] + yaw_noise(gen));
        z[1] = wrap(x[1] + yaw_noise(gen));
        z[2] += pos_noise(gen);

        // Kalman filter step
        u_k << u_applied;
        MatrixXd Kf = kalman_filter(x_hat, u_k, P, z, Q, R, H, F, G);
        
        //cout << x[0] << "," << x[1] << "," << x[2] << endl;
        std::vector<float> x_hat_std(x_hat.data(), x_hat.data() + x_hat.size());
        std::vector<float> z_std(z.data(), z.data() + z.size());
        std::vector<float> x_std(x.data(), x.data() + x.size());
        //cout << x_hat_std[0] << "," << x_hat_std[1] << "," << x_hat_std[2] << endl;
        //assert(false);
        if (count % steps_per_control == 0) {
            u = controller(x_hat_std, x_r);
            //u = controller(z_std, x_r);
            //u = controller(x_std, x_r);
            u[0] = std::clamp(u[0], -0.78539816f, 0.78539816f);
        }
        u_applied = u[0];     // remember for next iteration
 
        result.push_back({x[0], x[1], x[2], u[0]});
        kalman_result.push_back({x_hat_std[0], x_hat_std[1], x_hat_std[2], x[0], x[1], x[2], (float)Kf.trace(), (float)P.trace()});
        
        /*
        if (count == 0 || count == (int)trajectory.size() - 1) {
            cout << "Step " << count << ":" << endl;
            cout << "P trace: " << P.trace() << endl;
            cout << "Kf trace: " << Kf.trace() << endl;
            cout << "P matrix:\n" << P << endl;
            cout << "Kf matrix:\n" << Kf << endl;
        }
        */
        count += 1;
    }
    
    // Export debug data for Kalman filter
    std::ofstream out_kalman{"kalman.csv"};
    out_kalman << "x_hat_0" << "," << "x_hat_1" << "," << "x_hat_2" << "," << "x_0" << "," << "x_1" << "," << "x_2" << "," << "trace_Kf" << "," << "trace_P" << '\n';
    for (const auto& x_k: kalman_result) {
        if (x_k == kalman_result.back()) {
            out_kalman << x_k[0] << "," << x_k[1] << "," << x_k[2] << "," << x_k[3] << "," << x_k[4] << "," << x_k[5] << "," << x_k[6] << "," << x_k[7];
        }
        else {
            out_kalman << x_k[0] << "," << x_k[1] << "," << x_k[2] << "," << x_k[3] << "," << x_k[4] << "," << x_k[5] << "," << x_k[6] << "," << x_k[7] << '\n';
        }
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