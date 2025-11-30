// dyad_node.cpp
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <fftw3.h>
#include <serial/serial.h>   // use serial library (https://github.com/wjwwood/serial)
#include <thread>
#include <vector>
#include <deque>
#include <mutex>
#include <cmath>
#include <chrono>

using namespace std::chrono_literals;

class DyadNode : public rclcpp::Node {
public:
  DyadNode()
  : Node("dyad_node"),
    sample_rate_(1000),
    buffer_seconds_(180),
    window_s_(1.0),
    step_s_(0.1)
  {
    spec_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("dyad/spec", 10);
    phase_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("dyad/phase", 10);

    // configure serial (adjust port)
    try {
      serial_.setPort("/dev/ttyUSB0");
      serial_.setBaudrate(115200);
      serial_.setTimeout(serial::Timeout::simpleTimeout(1000));
      serial_.open();
    } catch(std::exception &e) {
      RCLCPP_WARN(this->get_logger(), "Serial open failed: %s", e.what());
    }

    // allocate buffer
    size_t buffer_len = sample_rate_ * buffer_seconds_;
    buffer_.resize(buffer_len, 0.0);
    buffer_mutex_ = std::make_unique<std::mutex>();

    // prepare FFTW (plan will be created per-window on demand)
    nfft_ = 1;
    while (nfft_ < (size_t)(window_s_ * sample_rate_)) nfft_ <<= 1;

    RCLCPP_INFO(this->get_logger(), "DyadNode initialized: sr=%d buffer=%d nfft=%d", sample_rate_, (int)buffer_len, (int)nfft_);

    // Start threads
    reader_thread_ = std::thread([this]() { this->serialReader(); });
    proc_thread_ = std::thread([this]() { this->processingLoop(); });
  }

  ~DyadNode() {
    if (reader_thread_.joinable()) reader_thread_.join();
    if (proc_thread_.joinable()) proc_thread_.join();
    if (serial_.isOpen()) serial_.close();
  }

private:
  void serialReader() {
    // read floats from serial line; expects space-separated values x,y,z
    std::string line;
    while (rclcpp::ok()) {
      try {
        line = serial_.readline(1024, "\n");
        if (line.empty()) { std::this_thread::sleep_for(5ms); continue; }
        std::stringstream ss(line);
        double a,b,c;
        if (!(ss >> a >> b >> c)) continue;
        double mag = sqrt(a*a + b*b + c*c);
        pushSample(mag);
      } catch (std::exception &e) {
        RCLCPP_WARN(this->get_logger(), "Serial read error: %s", e.what());
        std::this_thread::sleep_for(100ms);
      }
    }
  }

  void pushSample(double v) {
    std::lock_guard<std::mutex> lk(*buffer_mutex_);
    // circular push
    buffer_.push_back(v);
    if (buffer_.size() > (size_t)(sample_rate_ * buffer_seconds_)) {
      buffer_.pop_front();
    }
  }

  std::vector<double> snapshotBuffer() {
    std::lock_guard<std::mutex> lk(*buffer_mutex_);
    std::vector<double> out(buffer_.begin(), buffer_.end());
    return out;
  }

  void processingLoop() {
    size_t win = (size_t)(window_s_ * sample_rate_);
    size_t step = (size_t)(step_s_ * sample_rate_);
    while (rclcpp::ok()) {
      auto buf = snapshotBuffer();
      if (buf.size() < win) { std::this_thread::sleep_for(200ms); continue; }

      // take last buffer window and step through
      for (size_t offset = 0; offset + win <= buf.size(); offset += step) {
        // prepare window
        std::vector<double> seg(buf.begin() + offset, buf.begin() + offset + win);
        // demean
        double mean = 0.0;
        for (double s : seg) mean += s;
        mean /= seg.size();
        for (double &s : seg) s -= mean;

        // run FFTW (real-to-complex)
        size_t nfft = 1;
        while (nfft < win) nfft <<= 1;
        std::vector<double> in(nfft, 0.0);
        std::copy(seg.begin(), seg.end(), in.begin());
        fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (nfft/2 + 1));
        fftw_plan p = fftw_plan_dft_r2c_1d((int)nfft, in.data(), out, FFTW_ESTIMATE);
        fftw_execute(p);

        // compute amplitude at frequencies of interest (e.g., 11.6 Hz)
        double freq_res = (double)sample_rate_ / (double)nfft;
        int idx11 = (int)round(11.6 / freq_res);
        double amp11 = sqrt(out[idx11][0]*out[idx11][0] + out[idx11][1]*out[idx11][1]) / (win/2.0);

        // compute phase at 11.6
        double phase11 = atan2(out[idx11][1], out[idx11][0]);

        // publish amplitude & phase as Float32MultiArray (amplitude only for now)
        std_msgs::msg::Float32MultiArray amp_msg;
        amp_msg.data.push_back((float)amp11);
        spec_pub_->publish(amp_msg);

        std_msgs::msg::Float32MultiArray phase_msg;
        phase_msg.data.push_back((float)phase11);
        phase_pub_->publish(phase_msg);

        fftw_destroy_plan(p);
        fftw_free(out);
      } // end stepping windows

      std::this_thread::sleep_for( (int)(step * 1.0 / sample_rate_ * 1000)ms );
    } // end loop
  }

  // members
  serial::Serial serial_;
  std::deque<double> buffer_;
  std::unique_ptr<std::mutex> buffer_mutex_;
  int sample_rate_;
  int buffer_seconds_;
  double window_s_;
  double step_s_;
  size_t nfft_;
  std::thread reader_thread_;
  std::thread proc_thread_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr spec_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr phase_pub_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<DyadNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
