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
#include <thread>
#include <chrono>

#include "colors.h"

class MotorControllerNode : public rclcpp::Node {
public:
    MotorControllerNode() : Node("motor_controller") {
        // Declare parameters
        this->declare_parameter<std::string>("uart_port", "/dev/ttyACM0");
        this->declare_parameter<int>("baud_rate", 115200);
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
        bool enabled_temp = true;
        this->get_parameter("enabled", enabled_temp);
        enabled_ = enabled_temp;
        
        // UART file descriptor
        serial_fd_ = -1;
        connected_.store(false);
        rx_running_.store(false);
        arduino_seen_.store(false);
        last_cmd_time_ns_.store(0);
        last_rx_time_ns_.store(0);
        
        // Connect to Arduino (auto-detect /dev/ttyACM* if needed)
        if (enabled_) {
            const char* ports[] = {"/dev/ttyACM0", "/dev/ttyACM1"};
            bool connected = false;
            for (const char* p : ports) {
                if (connect(p, baud_rate)) {
                    RCLCPP_INFO(this->get_logger(), "Connected to Arduino on port: %s", p);
                    connected = true;
                    break;
                }
            }
            if (!connected) {
                RCLCPP_WARN(this->get_logger(), "Failed to connect to Arduino on ACM0/ACM1");
            }
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
    static int64_t nowNs() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }

    bool connect(const std::string& port, int baud_rate) {
        serial_fd_ = open(port.c_str(), O_RDWR | O_NOCTTY | O_SYNC | O_NONBLOCK);
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
        
        speed_t baud = B115200;
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
        // 0.1s read timeout (tenths of a second)
        tty.c_cc[VTIME] = 1;
        
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

        startRxThread();

        // Stop motors on startup
        usleep(100000);
        setMotors(0, 0);

        return true;
    }

    void disconnect() {
        stopRxThread();
        if (serial_fd_ >= 0) {
            setMotors(0, 0);
            close(serial_fd_);
            serial_fd_ = -1;
        }
        connected_.store(false);
        RCLCPP_INFO(this->get_logger(), "Disconnected");
    }

    void startRxThread() {
        if (rx_running_.load()) return;
        if (serial_fd_ < 0) return;
        rx_running_.store(true);
        rx_thread_ = std::thread([this]() { this->rxLoop(); });
    }

    void stopRxThread() {
        if (!rx_running_.load()) return;
        rx_running_.store(false);
        if (rx_thread_.joinable()) rx_thread_.join();
    }

    void rxLoop() {
        std::string buf;
        buf.reserve(512);

        while (rx_running_.load()) {
            if (serial_fd_ < 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }

            char tmp[256];
            errno = 0;
            ssize_t n = read(serial_fd_, tmp, sizeof(tmp));
            if (n > 0) {
                last_rx_time_ns_.store(nowNs());
                buf.append(tmp, tmp + n);

                size_t pos = 0;
                while (true) {
                    size_t nl = buf.find('\n', pos);
                    if (nl == std::string::npos) break;
                    std::string line = buf.substr(pos, nl - pos);
                    pos = nl + 1;
                    handleArduinoLine(line);
                }
                if (pos > 0) buf.erase(0, pos);
            } else if (n == 0) {
                // No data (non-blocking / timeout)
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            } else {
                const int e = errno;
                if (e != EAGAIN && e != EWOULDBLOCK) {
                    std::lock_guard<std::mutex> lk(rx_mutex_);
                    last_arduino_event_ = std::string("RX_ERROR: ") + strerror(e);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
    }

    static std::string trim(const std::string& s) {
        size_t a = 0;
        while (a < s.size() && (s[a] == '\r' || s[a] == '\n' || s[a] == ' ' || s[a] == '\t')) a++;
        size_t b = s.size();
        while (b > a && (s[b - 1] == '\r' || s[b - 1] == '\n' || s[b - 1] == ' ' || s[b - 1] == '\t')) b--;
        return s.substr(a, b - a);
    }

    void handleArduinoLine(const std::string& raw) {
        const std::string line = trim(raw);
        if (line.empty()) return;

        {
            std::lock_guard<std::mutex> lk(rx_mutex_);
            last_arduino_line_ = line;
            if (last_arduino_line_.size() > 160) {
                last_arduino_line_.resize(160);
            }
        }

        arduino_seen_.store(true);

        if (line.find("PING:OK") != std::string::npos) {
            // keep-alive from Arduino
            return;
        }
        if (line.find("EMERGENCY") != std::string::npos || line.find("TIMEOUT") != std::string::npos) {
            std::lock_guard<std::mutex> lk(rx_mutex_);
            last_arduino_event_ = line;
        }
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

        last_cmd_time_ns_.store(nowNs());
        last_cmd_linear_.store(static_cast<float>(msg->linear.x));
        last_cmd_angular_.store(static_cast<float>(msg->angular.z));
        
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
        const bool connected = connected_.load();
        const bool enabled = enabled_.load();

        const int64_t now = nowNs();
        const int64_t last_cmd = last_cmd_time_ns_.load();
        const int64_t last_rx = last_rx_time_ns_.load();

        const int64_t cmd_age_ms = (last_cmd > 0) ? (now - last_cmd) / 1000000 : -1;
        const int64_t rx_age_ms = (last_rx > 0) ? (now - last_rx) / 1000000 : -1;

        std::string line;
        std::string event;
        {
            std::lock_guard<std::mutex> lk(rx_mutex_);
            line = last_arduino_line_;
            event = last_arduino_event_;
        }

        // JSON (manual, minimal escaping for quotes/backslashes)
        auto esc = [](const std::string& s) {
            std::string o;
            o.reserve(s.size());
            for (char c : s) {
                if (c == '\\' || c == '"') o.push_back('\\');
                if (static_cast<unsigned char>(c) >= 0x20) o.push_back(c);
            }
            return o;
        };

        std::string json = "{";
        json += "\"connected\":" + std::string(connected ? "true" : "false");
        json += ",\"enabled\":" + std::string(enabled ? "true" : "false");
        json += ",\"left_speed\":" + std::to_string(last_left_speed_.load());
        json += ",\"right_speed\":" + std::to_string(last_right_speed_.load());
        json += ",\"last_cmd_age_ms\":" + std::to_string(cmd_age_ms);
        json += ",\"last_rx_age_ms\":" + std::to_string(rx_age_ms);
        json += ",\"arduino_seen\":" + std::string(arduino_seen_.load() ? "true" : "false");
        json += ",\"last_cmd_linear\":" + std::to_string(last_cmd_linear_.load());
        json += ",\"last_cmd_angular\":" + std::to_string(last_cmd_angular_.load());
        json += ",\"arduino_last_line\":\"" + esc(line) + "\"";
        json += ",\"arduino_last_event\":\"" + esc(event) + "\"";
        json += "}";

        status_msg.data = json;
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

    // Arduino RX telemetry
    std::atomic<bool> rx_running_{false};
    std::thread rx_thread_;
    std::mutex rx_mutex_;
    std::string last_arduino_line_;
    std::string last_arduino_event_;
    std::atomic<bool> arduino_seen_{false};
    std::atomic<int64_t> last_cmd_time_ns_{0};
    std::atomic<int64_t> last_rx_time_ns_{0};
    std::atomic<float> last_cmd_linear_{0.0f};
    std::atomic<float> last_cmd_angular_{0.0f};
    
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
