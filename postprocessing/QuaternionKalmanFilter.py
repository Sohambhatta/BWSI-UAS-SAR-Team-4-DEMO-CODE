import numpy as np
import quaternion  # pip install numpy-quaternion
import os
import csv

# your_quats = []
# pos_csv_path = "DATAS/July29Flight2BII10/pos.csv"

# with open(pos_csv_path, "r") as f:
#     reader = csv.reader(f)
#     for row in reader:
#         if len(row) > 0 and row[0].strip().lower() == 'frame':
#             break
#     for row in reader:
#         if len(row) < 9 or row[6] == '' or row[7] == '' or row[8] == '':
#             continue
#         frame, time, i, j, k, r, x, y, z = row[:9]
#         your_quats.append([float(r), float(i), float(j), float(k)])

# your_quast = np.array(your_quats, dtype=np.float64)

class QuaternionESKF:
    def __init__(self, process_noise=1e-4, measurement_noise=1e-2, dt=1/360):
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.reset()
    
    def reset(self):
        self.q = quaternion.quaternion(1, 0, 0, 0)
        self.P = np.eye(3) * 1e-3
        self.angular_velocity = np.array([0.0, 0.0, 0.0])
        self.initialized = False
    
    def ensure_consistent_quaternion_sign(self, q_new, q_reference):
        """
        Ensure quaternion has consistent sign relative to reference.
        This handles the double cover q ≡ -q properly.
        """
        q_new_normalized = q_new.normalized()
        q_ref_normalized = q_reference.normalized()
        
        # Check dot product to determine if we should flip sign
        dot_product = (q_new_normalized.w * q_ref_normalized.w + 
                      q_new_normalized.x * q_ref_normalized.x + 
                      q_new_normalized.y * q_ref_normalized.y + 
                      q_new_normalized.z * q_ref_normalized.z)
        
        # If dot product is negative, the quaternions are on opposite hemispheres
        # Flip to ensure we take the shorter path
        if dot_product < 0:
            return quaternion.quaternion(-q_new_normalized.w, 
                                       -q_new_normalized.x,
                                       -q_new_normalized.y, 
                                       -q_new_normalized.z)
        else:
            return q_new_normalized
    
    def minimal_quaternion_difference(self, q_target, q_current):
        """
        Compute the minimal rotation difference between two quaternions.
        This properly handles the double cover and returns the rotation
        in the tangent space (rotation vector).
        """
        q_target_consistent = self.ensure_consistent_quaternion_sign(q_target, q_current)
        
        # Compute error quaternion: q_error = q_target * q_current^(-1)
        q_error = q_target_consistent * q_current.conj()
        
        # Convert to rotation vector (this is minimal and handles the manifold properly)
        error_rotation_vector = quaternion.as_rotation_vector(q_error)
        
        return error_rotation_vector, q_target_consistent
    
    def skew_symmetric(self, v):
        return np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])
    
    def predict(self):
        if not self.initialized:
            return
        
        # Propagate quaternion
        omega_dt = self.angular_velocity * self.dt
        if np.linalg.norm(omega_dt) > 1e-8:
            dq = quaternion.from_rotation_vector(omega_dt)
            self.q = self.q * dq
            # Note: No explicit normalization needed, numpy-quaternion handles this
        
        # Propagate error covariance
        omega_norm = np.linalg.norm(self.angular_velocity)
        if omega_norm > 1e-8:
            dt_omega = omega_norm * self.dt
            sin_dt = np.sin(dt_omega)
            cos_dt = np.cos(dt_omega)
            
            Omega = self.skew_symmetric(self.angular_velocity)
            F = np.eye(3) * cos_dt + (Omega / omega_norm) * sin_dt + \
                (Omega @ Omega / (omega_norm**2)) * (1 - cos_dt)
        else:
            F = np.eye(3) + self.skew_symmetric(self.angular_velocity) * self.dt
        
        Q = np.eye(3) * self.process_noise * self.dt**2
        self.P = F @ self.P @ F.T + Q
    
    def update(self, measurement_array):
        measurement = quaternion.quaternion(*measurement_array).normalized()
        
        if not self.initialized:
            self.q = measurement
            self.initialized = True
            return quaternion.as_float_array(self.q), False, 0.0
        
        # Use the corrected minimal difference function
        innovation, measurement_consistent = self.minimal_quaternion_difference(measurement, self.q)
        innovation_angle = np.linalg.norm(innovation)
        
        # Outlier detection based on innovation in rotation vector space
        # This properly handles the double cover since we're working in tangent space
        is_outlier = innovation_angle > np.pi/6  # 30 degrees
        
        if not is_outlier:
            H = np.eye(3)
            R = np.eye(3) * self.measurement_noise
            
            S = H @ self.P @ H.T + R
            K = self.P @ H.T @ np.linalg.inv(S)
            
            # Update error state
            error_correction = K @ innovation
            
            # Apply correction in rotation vector space
            correction_quat = quaternion.from_rotation_vector(error_correction)
            self.q = correction_quat * self.q
            
            # Update covariance
            I_KH = np.eye(3) - K @ H
            self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
            
            # Update angular velocity estimate
            if hasattr(self, 'last_innovation'):
                velocity_update = (innovation - self.last_innovation) / self.dt
                alpha = 0.1
                self.angular_velocity = (1-alpha) * self.angular_velocity + alpha * velocity_update
            
            self.last_innovation = innovation
        else:
            # For outliers, temporarily increase process noise
            self.P += np.eye(3) * self.process_noise * 10
        
        return quaternion.as_float_array(self.q), is_outlier, innovation_angle

def test_double_cover_handling():
    """
    Test that the filter properly handles quaternion double cover
    """
    # Create test data with intentional sign flips that represent the same rotation
    test_quats = []
    
    # Small rotations around identity - these should flip w sign legitimately
    angles = np.linspace(-0.1, 0.1, 20)  # Small angles around 0
    for angle in angles:
        q = quaternion.from_rotation_vector([angle, 0, 0])
        # Randomly flip sign to test double cover handling
        if np.random.random() > 0.5:
            q = quaternion.quaternion(-q.w, -q.x, -q.y, -q.z)
        test_quats.append([q.w, q.x, q.y, q.z])
    
    test_quats = np.array(test_quats)
    
    # Apply filter
    filtered, outliers, innovations = eskf_filter_quaternions(test_quats, dt=1/20)
    
    print(f"Input w values: {test_quats[:5, 0]}")
    print(f"Filtered w values: {filtered[:5, 0]}")
    print(f"Outliers detected: {sum(outliers)} out of {len(test_quats)}")
    print(f"Max innovation angle: {np.max(innovations):.4f} radians")
    
    return test_quats, filtered, outliers, innovations

def eskf_filter_quaternions(quats, dt=1/360, process_noise=1e-3, measurement_noise=5e-3):
    """
    Apply Error-State Kalman Filter to quaternion sequence
    
    Args:
        quats: Array of quaternions as numpy arrays (N x 4) in [w,x,y,z] format
        dt: Time step between samples
        process_noise: Process noise variance
        measurement_noise: Measurement noise variance
    
    Returns:
        filtered_quats: Filtered quaternions (N x 4)
        outlier_flags: Boolean array indicating outliers
        innovation_angles: Angular innovation at each step
    """
    eskf = QuaternionESKF(process_noise=process_noise, 
                         measurement_noise=measurement_noise, 
                         dt=dt)
    
    filtered_quats = []
    outlier_flags = []
    innovation_angles = []
    
    for quat_array in quats:
        eskf.predict()
        filtered_quat, is_outlier, innovation_angle = eskf.update(quat_array)
        
        filtered_quats.append(filtered_quat.copy())
        outlier_flags.append(is_outlier)
        innovation_angles.append(innovation_angle)
    
    return np.array(filtered_quats), outlier_flags, innovation_angles

# Enhanced analysis functions
def analyze_quaternion_motion(quats, dt=1/360):
    """Analyze quaternion motion using numpy-quaternion tools"""
    angular_velocities = []
    accelerations = []
    
    for i in range(1, len(quats)):
        q1 = np.quaternion(*quats[i-1]).normalized()
        q2 = np.quaternion(*quats[i]).normalized()
        
        # Compute angular velocity
        q_diff = q2 * q1.conj()
        rot_vec = quaternion.as_rotation_vector(q_diff)
        angular_vel = rot_vec / dt
        angular_velocities.append(angular_vel)
        
        # Compute angular acceleration
        if len(angular_velocities) > 1:
            accel = (angular_velocities[-1] - angular_velocities[-2]) / dt
            accelerations.append(accel)
    
    return np.array(angular_velocities), np.array(accelerations)



# filtered_quats, outliers, innovations = eskf_filter_quaternions(
#     your_quats, 
#     dt=1/360,
#     process_noise=1e-3,
#     measurement_noise=5e-3
# )

# # Analyze motion
# angular_vels, angular_accels = analyze_quaternion_motion(filtered_quats, dt=1/360)

# Visualization
# import matplotlib.pyplot as plt

# fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# # Plot quaternion components
# axes[0,0].plot(your_quats, alpha=0.7, label=['w', 'x', 'y', 'z'])
# axes[0,0].plot(filtered_quats, '--', label=['w_filt', 'x_filt', 'y_filt', 'z_filt'])
# axes[0,0].set_title('Quaternion Components')
# axes[0,0].legend()

# # Plot innovations and outliers
# axes[0,1].plot(innovations, label='Innovation Angle')
# outlier_indices = np.where(outliers)[0]
# axes[0,1].scatter(outlier_indices, np.array(innovations)[outlier_indices], 
#                  color='red', s=50, label='Outliers', zorder=5)
# axes[0,1].set_title('Innovation Analysis')
# axes[0,1].set_ylabel('Radians')
# axes[0,1].legend()

# # Plot angular velocities
# if len(angular_vels) > 0:
#     axes[1,0].plot(np.linalg.norm(angular_vels, axis=1), label='Angular Speed')
#     axes[1,0].set_title('Angular Velocity Magnitude')
#     axes[1,0].set_ylabel('rad/s')
#     axes[1,0].legend()

# # Plot angular accelerations
# if len(angular_accels) > 0:
#     axes[1,1].plot(np.linalg.norm(angular_accels, axis=1), label='Angular Acceleration')
#     axes[1,1].set_title('Angular Acceleration Magnitude')
#     axes[1,1].set_ylabel('rad/s²')
#     axes[1,1].legend()

# plt.tight_layout()
# plt.show()

# print(f"Detected {sum(outliers)} outliers out of {len(your_quats)} samples")
# print(f"Max angular velocity: {np.max(np.linalg.norm(angular_vels, axis=1)):.2f} rad/s")
