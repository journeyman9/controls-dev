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

vector<float> get_coefficients(float y0, float y1, float yd0, float yd1) {
    float a = y0;
    float b = yd0;
    float c = 3 * (y1 - y0) - 2 * yd0 - yd1;
    float d = 2 * (y0 - y1) + yd0 + yd1;
    return {a, b, c, d};
}

vector<vector<float>> reference_trajectory_generation(float dt, float T, float S1, float S2, float S3) {
    /*
    y(t) = a + bt + ct^2 + dt^3
    yd(t) = b + 2ct + 3dt^2
    
    with constraints for psi_1
    y(0) = 1, y(1) = 0
    yd(0) = 0, yd(1) = 0
    
    with constraints for psi_2
    y(0) = 1, y(1) = 0
    yd(0) = 0, yd(1) = 0
    
    with constraints for y2
    y(0) = -1, y(1) = 0
    yd(0) = 0, yd(1) = 0
    */
    
    vector<vector<float>> x_r_t;
    int steps = static_cast<int>(T / dt);

    // Coefficients for psi_1
    vector<float> coeffs_psi1 = get_coefficients(1, 0, 0, 0);
    // Coefficients for psi_2
    vector<float> coeffs_psi2 = get_coefficients(1, 0, 0, 0);
    // Coefficients for y2
    vector<float> coeffs_y2 = get_coefficients(-1, 0, 0, 0);

    float t, y1, yd1, y2, yd2, y3, yd3;
    for (int i = 0; i <= steps; ++i) {
        t = i * dt / T;
        
        // Calculate y and yd for psi_1
        y1 = coeffs_psi1[0] + coeffs_psi1[1] * t + coeffs_psi1[2] * pow(t, 2) + coeffs_psi1[3] * pow(t, 3);
        yd1 = coeffs_psi1[1] + 2 * coeffs_psi1[2] * t + 3 * coeffs_psi1[3] * pow(t, 2);
        
        // Calculate y and yd for psi_2
        y2 = coeffs_psi2[0] + coeffs_psi2[1] * t + coeffs_psi2[2] * pow(t, 2) + coeffs_psi2[3] * pow(t, 3);
        yd2 = coeffs_psi2[1] + 2 * coeffs_psi2[2] * t + 3 * coeffs_psi2[3] * pow(t, 2);
        
        // Calculate y and yd for y2
        y3 = coeffs_y2[0] + coeffs_y2[1] * t + coeffs_y2[2] * pow(t, 2) + coeffs_y2[3] * pow(t, 3);
        yd3 = coeffs_y2[1] + 2 * coeffs_y2[2] * t + 3 * coeffs_y2[3] * pow(t, 2);

        // Scale the results
        y1 *= S1;
        yd1 *= S1;
        y2 *= S2;
        yd2 *= S2;
        y3 *= S3;
        yd3 *= S3;
        x_r_t.push_back({y1, y2, y3, yd1, yd2, yd3});
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

    // derivative set to 0
    MatrixXd A = MatrixXd::Constant(3, 3, 0.0);
    A(1, 0) = -0.19740973;
    A(1, 1) = 0.19740973;
    A(2, 1) = -2.0120;

    VectorXd B(3);
    B << -0.35052265, -0.00997366, 0;

    MatrixXd W = MatrixXd::Identity(3, 3);
    //W(0, 0) = 1.0 / pow(1.0472, 2);
    W(0, 0) = 0.0;
    W(1, 1) = 1.0 / pow(0.0872, 2);
    W(2, 2) = 1.0 / pow(0.15, 2);

    VectorXd x_rr = VectorXd(3);
    x_rr << x_r[0], x_r[1], x_r[2];

    VectorXd xd_rr = VectorXd(3);
    xd_rr << x_r[3], x_r[4], x_r[5];

    VectorXd result = (B.transpose() * W.transpose() * W * B).inverse() * B.transpose() * W.transpose() * W * (-A * x_rr + xd_rr);

    u[0] = u_fb[0] + result(0);


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
    vector<float> x = {1*S1, 1*S2, -1 * S3}; // Initial state
    
    vector<vector<float>> trajectory = reference_trajectory_generation(dt, T, S1, S2, S3);

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