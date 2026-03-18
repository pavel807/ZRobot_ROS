#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <errno.h>
#include <cstring>
#include <string>
#include <memory>
#include <mutex>
#include <atomic>

#include "colors.h"

class MotorControllerNode : public rclcpp::Node {
public:
    MotorControllerNode() : Node("motor_controller") {
        // Declare parameters
        this->declare_parameter<std::string>("uart_port", "/dev/ttyACM0");
        this->declare_parameter<int>("baud_rate", 9600);
        this->declare_parameter<int>("max_speed", 245);
        this->declare_parameter<int>("min_speed", 165);
        this->declare_parameter<bool>("enabled", true);
        
        // Get parameters
        std::string uart_port;
        int baud_rate;
        
        this->get_parameter("uart_port", uart_port);
        this->get_parameter("baud_rate", baud_rate);
        this->get_parameter("max_speed", max_speed_);
        this->get_parameter("min_speed", min_speed_);
        bool enabled_temp;
        this->get_parameter("enabled", enabled_temp);
        enabled_ = enabled_temp;
        
        // UART file descriptor
        serial_fd_ = -1;
        connected_.store(false);
        
        // Connect to Arduino
        if (enabled_) {
            connect(uart_port, baud_rate);
        }
        
        // Create subscription for velocity commands (from obstacle avoidance)
        cmd_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "cmd_vel_safe", 10,
            std::bind(&MotorControllerNode::cmdVelCallback, this, std::placeholders::_1)
        );
        
        // Create subscription for motor enable/disable
        enable_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            "motor_enable", 10,
            std::bind(&MotorControllerNode::enableCallback, this, std::placeholders::_1)
        );
        
        // Create status publisher
        status_pub_ = this->create_publisher<std_msgs::msg::String>("motor_status", 10);
        
        // Create timer for status updates
        status_timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&MotorControllerNode::publishStatus, this)
        );
        
        last_left_speed_ = 0;
        last_right_speed_ = 0;
        
        RCLCPP_INFO(this->get_logger(), "Motor Controller node started");
    }
    
    ~MotorControllerNode() {
        disconnect();
    }

private:
    bool connect(const std::string& port, int baud_rate) {
        serial_fd_ = open(port.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
        if (serial_fd_ < 0) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open %s: %s", port.c_str(), strerror(errno));
            connected_.store(false);
            return false;
        }

        struct termios tty;
        memset(&tty, 0, sizeof(tty));
        if (tcgetattr(serial_fd_, &tty) != 0) {
            RCLCPP_ERROR(this->get_logger(), "Error from tcgetattr: %s", strerror(errno));
            close(serial_fd_);
            serial_fd_ = -1;
            connected_.store(false);
            return false;
        }
        
        speed_t baud = B9600;
        if (baud_rate == 115200) baud = B115200;
        else if (baud_rate == 57600) baud = B57600;
        else if (baud_rate == 19200) baud = B19200;
        
        cfsetospeed(&tty, baud);
        cfsetispeed(&tty, baud);
        
        tty.c_cflag &= ~PARENB;
        tty.c_cflag &= ~CSTOPB;
        tty.c_cflag &= ~CSIZE;
        tty.c_cflag |= CS8;
        tty.c_cflag |= CREAD | CLOCAL;
        
        tty.c_lflag &= ~ICANON;
        tty.c_lflag &= ~ECHO;
        tty.c_lflag &= ~ECHOE;
        tty.c_lflag &= ~ECHONL;
        tty.c_lflag &= ~ISIG;
        
        tty.c_iflag &= ~(IXON | IXOFF | IXANY);
        tty.c_iflag &= ~(IGNBRK|BRKINT|PARMRK|ISTRIP|INLCR|IGNCR|ICRNL);
        
        tty.c_oflag &= ~OPOST;
        tty.c_oflag &= ~ONLCR;
        
        tty.c_cc[VMIN] = 0;
        tty.c_cc[VTIME] = 10;
        
        if (tcsetattr(serial_fd_, TCSANOW, &tty) != 0) {
            RCLCPP_ERROR(this->get_logger(), "Error from tcsetattr: %s", strerror(errno));
            close(serial_fd_);
            serial_fd_ = -1;
            connected_.store(false);
            return false;
        }
        
        tcflush(serial_fd_, TCIOFLUSH);
        
        connected_.store(true);
        RCLCPP_INFO(this->get_logger(), "Connected to %s @ %d baud", port.c_str(), baud_rate);

        // Stop motors on startup
        usleep(100000);
        setMotors(0, 0);

        return true;
    }

    void disconnect() {
        if (serial_fd_ >= 0) {
            setMotors(0, 0);
            close(serial_fd_);
            serial_fd_ = -1;
        }
        connected_.store(false);
        RCLCPP_INFO(this->get_logger(), "Disconnected");
    }
    
    void setMotors(int left_speed, int right_speed) {
        if (!connected_.load() || serial_fd_ < 0) return;
        
        left_speed = std::max(0, std::min(255, left_speed));
        right_speed = std::max(0, std::min(255, right_speed));
        
        std::lock_guard<std::mutex> lock(uart_mutex_);
        
        std::string cmd_l = "/motor L " + std::to_string(left_speed) + "\n";
        std::string cmd_r = "/motor R " + std::to_string(right_speed) + "\n";
        
        ssize_t written = write(serial_fd_, cmd_l.c_str(), cmd_l.length());
        if (written < 0) {
            RCLCPP_ERROR(this->get_logger(), "Write error: %s", strerror(errno));
            connected_.store(false);
            return;
        }

        usleep(5000);

        written = write(serial_fd_, cmd_r.c_str(), cmd_r.length());
        if (written < 0) {
            RCLCPP_ERROR(this->get_logger(), "Write error: %s", strerror(errno));
            connected_.store(false);
            return;
        }
        
        tcdrain(serial_fd_);
    }
    
    void cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg) {
        bool enabled = enabled_.load();
        bool connected = connected_.load();
        if (!enabled || !connected) return;
        
        // Convert Twist to motor speeds
        // linear.x = forward/backward speed
        // angular.z = rotation speed
        
        float linear = msg->linear.x;
        float angular = msg->angular.z;
        
        // Differential drive kinematics
        int left_speed = (int)((linear - angular) * max_speed_);
        int right_speed = (int)((linear + angular) * max_speed_);
        
        // Apply minimum speed threshold
        if (left_speed > 0 && left_speed < min_speed_) left_speed = min_speed_;
        if (right_speed > 0 && right_speed < min_speed_) right_speed = min_speed_;
        
        setMotors(left_speed, right_speed);
        
        last_left_speed_ = left_speed;
        last_right_speed_ = right_speed;
        
        RCLCPP_DEBUG(this->get_logger(), "CmdVel: L=%d R=%d", left_speed, right_speed);
    }
    
    void enableCallback(const std_msgs::msg::Bool::SharedPtr msg) {
        bool new_enabled = msg->data;
        enabled_ = new_enabled;
        if (!enabled_) {
            setMotors(0, 0);
            RCLCPP_INFO(this->get_logger(), "Motors disabled");
        } else {
            RCLCPP_INFO(this->get_logger(), "Motors enabled");
        }
    }
    
    void publishStatus() {
        if (status_pub_->get_subscription_count() == 0) return;
        
        std_msgs::msg::String status_msg;
        char status_buf[128];
        snprintf(status_buf, sizeof(status_buf),
                "{\"connected\":%s,\"enabled\":%s,\"left_speed\":%d,\"right_speed\":%d}",
                connected_.load() ? "true" : "false",
                enabled_.load() ? "true" : "false",
                last_left_speed_.load(),
                last_right_speed_.load());
        status_msg.data = status_buf;
        status_pub_->publish(status_msg);
    }
    
    // ROS2 members
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr enable_sub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
    rclcpp::TimerBase::SharedPtr status_timer_;
    
    // UART
    int serial_fd_;
    std::mutex uart_mutex_;
    std::atomic<bool> connected_;
    
    // Parameters
    int max_speed_;
    int min_speed_;
    std::atomic<bool> enabled_{true};
    std::atomic<int> last_left_speed_{0};
    std::atomic<int> last_right_speed_{0};
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MotorControllerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
