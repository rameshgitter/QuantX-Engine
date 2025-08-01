#include "onnx_predictor.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <nlohmann/json.hpp>

namespace quantx {

class LOBFeatures {
public:
    double spread = 0.0;
    double volume_imbalance = 0.0;
    double bid_ask_ratio = 0.0;
    double depth_imbalance = 0.0;
    double relative_spread = 0.0;
    double price_impact_bid = 0.0;
    double price_impact_ask = 0.0;
    double micro_price_diff = 0.0;
    double volume_ratio = 0.0;
    double order_flow = 0.0;
    double spread_change = 0.0;
    double volume_imbalance_change = 0.0;
    double spread_momentum = 0.0;
    double volume_imbalance_momentum = 0.0;

    std::vector<double> toVector() const {
        return {
            spread, volume_imbalance, bid_ask_ratio, depth_imbalance,
            relative_spread, price_impact_bid, price_impact_ask,
            micro_price_diff, volume_ratio, order_flow,
            spread_change, volume_imbalance_change,
            spread_momentum, volume_imbalance_momentum
        };
    }
};

class PredictionResult {
public:
    std::chrono::high_resolution_clock::time_point timestamp;
    std::vector<double> probabilities;
    int direction = 0;
    double confidence = 0.0;
    double signal_strength = 0.0;
};

class OrderBook {
public:
    std::vector<Order> bids;
    std::vector<Order> asks;
    double spread = 0.0;
    double mid_price = 0.0;
};

class Order {
public:
    double price = 0.0;
    int quantity = 0;
};

class IVPrediction {
public:
    std::chrono::high_resolution_clock::time_point timestamp;
    double implied_volatility = 0.0;
    double confidence = 0.0;
};

class ONNXPredictor {
private:
    Ort::Env* env_;
    Ort::Session* session_;
    std::vector<std::string> input_names_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::string> output_names_;
    std::vector<std::vector<int64_t>> output_shapes_;
    Ort::MemoryInfo memory_info_;
    bool initialized_;
    std::string model_path_;

    void extractModelInfo() {
        // Get input info
        size_t num_inputs = session_->GetInputCount();
        input_names_.clear();
        input_shapes_.clear();

        for (size_t i = 0; i < num_inputs; ++i) {
            auto input_name = session_->GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
            input_names_.push_back(input_name.get());

            auto input_shape = session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            input_shapes_.push_back(input_shape);
        }

        // Get output info
        size_t num_outputs = session_->GetOutputCount();
        output_names_.clear();
        output_shapes_.clear();

        for (size_t i = 0; i < num_outputs; ++i) {
            auto output_name = session_->GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
            output_names_.push_back(output_name.get());

            auto output_shape = session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            output_shapes_.push_back(output_shape);
        }
    }

public:
    ONNXPredictor() 
        : env_(nullptr), session_(nullptr), initialized_(false) {
        memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    }

    ~ONNXPredictor() {
        delete session_;
        delete env_;
    }

    bool initialize(const std::string& model_path) {
        try {
            model_path_ = model_path;

            // Create ONNX Runtime environment
            env_ = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "QuantXEngine");

            // Create session options
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

            // Create session
            session_ = new Ort::Session(*env_, model_path.c_str(), session_options);

            // Extract model information
            extractModelInfo();

            initialized_ = true;
            std::cout << "ONNX model loaded: " << model_path << std::endl;

            return true;

        } catch (const std::exception& e) {
            std::cerr << "Failed to initialize ONNX model: " << e.what() << std::endl;
            initialized_ = false;
            return false;
        }
    }

    std::vector<float> predict(const std::vector<float>& input) {
        if (!initialized_) {
            throw std::runtime_error("ONNX predictor not initialized");
        }

        std::lock_guard<std::mutex> lock(prediction_mutex_);

        try {
            // Prepare input tensor
            std::vector<Ort::Value> input_tensors;
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                memory_info_, 
                const_cast<float*>(input.data()), 
                input.size(),
                input_shapes_[0].data(), 
                input_shapes_[0].size()
            ));

            // Prepare input names
            std::vector<const char*> input_names_cstr;
            for (const auto& name : input_names_) {
                input_names_cstr.push_back(name.c_str());
            }

            // Prepare output names
            std::vector<const char*> output_names_cstr;
            for (const auto& name : output_names_) {
                output_names_cstr.push_back(name.c_str());
            }

            // Run inference
            auto output_tensors = session_->Run(
                Ort::RunOptions{nullptr},
                input_names_cstr.data(),
                input_tensors.data(),
                input_names_cstr.size(),
                output_names_cstr.data(),
                output_names_cstr.size()
            );

            // Extract output
            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

            return std::vector<float>(output_data, output_data + output_size);

        } catch (const std::exception& e) {
            std::cerr << "ONNX prediction error: " << e.what() << std::endl;
            return {};
        }
    }

    bool isInitialized() const {
        return initialized_;
    }

private:
    std::mutex prediction_mutex_;
};

class LOBPredictor {
private:
    std::unique_ptr<ONNXPredictor> predictor_;
    std::vector<std::vector<float>> feature_buffer_;
    size_t buffer_index_;
    bool buffer_full_;
    nlohmann::json normalization_params_;
    static constexpr size_t SEQUENCE_LENGTH = 10;
    static constexpr size_t FEATURE_COUNT = 14;

    std::vector<float> normalizeFeatures(const std::vector<float>& features) {
        if (!normalization_params_.contains("means") || !normalization_params_.contains("stds")) {
            return features;  // Return unnormalized if params not available
        }

        auto means = normalization_params_["means"];
        auto stds = normalization_params_["stds"];

        std::vector<float> normalized(features.size());
        for (size_t i = 0; i < features.size() && i < means.size() && i < stds.size(); ++i) {
            double mean = means[i];
            double std = stds[i];
            normalized[i] = static_cast<float>((features[i] - mean) / (std + 1e-8));
        }

        return normalized;
    }

    std::vector<float> extractFeatures(const OrderBook& order_book) {
        std::vector<float> features(FEATURE_COUNT, 0.0f);

        if (order_book.bids.empty() || order_book.asks.empty()) {
            return features;
        }

        // Basic features
        features[0] = static_cast<float>(order_book.spread);  // spread

        // Volume imbalance
        double bid_volume = 0.0, ask_volume = 0.0;
        for (size_t i = 0; i < std::min(size_t(5), order_book.bids.size()); ++i) {
            bid_volume += order_book.bids[i].quantity;
        }
        for (size_t i = 0; i < std::min(size_t(5), order_book.asks.size()); ++i) {
            ask_volume += order_book.asks[i].quantity;
        }

        features[1] = static_cast<float>((bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-8));  // volume_imbalance
        features[2] = static_cast<float>(bid_volume / (ask_volume + 1e-8));  // bid_ask_ratio

        // Depth imbalance (weighted by distance from mid)
        double weighted_bid_depth = 0.0, weighted_ask_depth = 0.0;
        for (size_t i = 0; i < std::min(size_t(10), order_book.bids.size()); ++i) {
            double weight = 1.0 / (1.0 + i);
            weighted_bid_depth += order_book.bids[i].quantity * weight;
        }
        for (size_t i = 0; i < std::min(size_t(10), order_book.asks.size()); ++i) {
            double weight = 1.0 / (1.0 + i);
            weighted_ask_depth += order_book.asks[i].quantity * weight;
        }

        features[3] = static_cast<float>((weighted_bid_depth - weighted_ask_depth) / 
                                       (weighted_bid_depth + weighted_ask_depth + 1e-8));  // depth_imbalance

        features[4] = static_cast<float>(order_book.spread / order_book.mid_price);  // relative_spread

        // Price impact estimation
        features[5] = static_cast<float>(1000.0 / bid_volume);  // price_impact_bid
        features[6] = static_cast<float>(1000.0 / ask_volume);  // price_impact_ask

        // Micro price difference
        double micro_price = (order_book.bids[0].price * ask_volume + order_book.asks[0].price * bid_volume) / 
                            (bid_volume + ask_volume);
        features[7] = static_cast<float>((micro_price - order_book.mid_price) / order_book.mid_price);  // micro_price_diff

        // Volume ratio
        features[8] = static_cast<float>(bid_volume / (bid_volume + ask_volume));  // volume_ratio

        // Order flow (simplified)
        features[9] = static_cast<float>((bid_volume - ask_volume) / 1000.0);  // order_flow

        // Temporal features (will be calculated in updateFeatureBuffer)
        features[10] = 0.0f;  // spread_change
        features[11] = 0.0f;  // volume_imbalance_change
        features[12] = 0.0f;  // spread_momentum
        features[13] = 0.0f;  // volume_imbalance_momentum

        return features;
    }

    void updateFeatureBuffer(const std::vector<float>& features) {
        std::lock_guard<std::mutex> lock(buffer_mutex_);

        // Calculate temporal features if we have previous data
        if (buffer_index_ > 0 || buffer_full_) {
            size_t prev_idx = (buffer_index_ - 1 + SEQUENCE_LENGTH) % SEQUENCE_LENGTH;
            auto& prev_features = feature_buffer_[prev_idx];

            // Calculate changes
            std::vector<float> updated_features = features;
            updated_features[10] = features[0] - prev_features[0];  // spread_change
            updated_features[11] = features[1] - prev_features[1];  // volume_imbalance_change

            // Calculate momentum if we have enough history
            if ((buffer_index_ > 1) || (buffer_full_ && buffer_index_ != 1)) {
                size_t prev_prev_idx = (buffer_index_ - 2 + SEQUENCE_LENGTH) % SEQUENCE_LENGTH;
                auto& prev_prev_features = feature_buffer_[prev_prev_idx];

                updated_features[12] = updated_features[10] - (prev_features[0] - prev_prev_features[0]);  // spread_momentum
                updated_features[13] = updated_features[11] - (prev_features[1] - prev_prev_features[1]);  // volume_imbalance_momentum
            }

            feature_buffer_[buffer_index_] = updated_features;
        } else {
            feature_buffer_[buffer_index_] = features;
        }

        buffer_index_ = (buffer_index_ + 1) % SEQUENCE_LENGTH;
        if (buffer_index_ == 0) {
            buffer_full_ = true;
        }
    }

    std::vector<float> getSequenceInput() {
        std::lock_guard<std::mutex> lock(buffer_mutex_);

        std::vector<float> sequence_input;
        sequence_input.reserve(SEQUENCE_LENGTH * FEATURE_COUNT);

        if (!buffer_full_) {
            // If buffer not full, pad with zeros and use available data
            for (int i = 0; i < SEQUENCE_LENGTH; ++i) {
                if (i < static_cast<int>(buffer_index_)) {
                    for (float feature : feature_buffer_[i]) {
                        sequence_input.push_back(feature);
                    }
                } else {
                    for (int j = 0; j < FEATURE_COUNT; ++j) {
                        sequence_input.push_back(0.0f);
                    }
                }
            }
        } else {
            // Use circular buffer in correct order
            for (int i = 0; i < SEQUENCE_LENGTH; ++i) {
                size_t idx = (buffer_index_ + i) % SEQUENCE_LENGTH;
                for (float feature : feature_buffer_[idx]) {
                    sequence_input.push_back(feature);
                }
            }
        }

        return sequence_input;
    }

    PredictionResult interpretPrediction(const std::vector<float>& output) {
        PredictionResult result;
        result.timestamp = std::chrono::high_resolution_clock::now();

        if (output.size() < 3) {
            return result;
        }

        // Assuming output is [sell_prob, neutral_prob, buy_prob]
        result.probabilities = {output[0], output[1], output[2]};

        // Find the class with highest probability
        auto max_it = std::max_element(output.begin(), output.end());
        int predicted_class = static_cast<int>(std::distance(output.begin(), max_it));

        result.confidence = *max_it;

        // Convert to signal strength and direction
        if (predicted_class == 0) {  // Sell
            result.direction = -1;
            result.signal_strength = -output[0];
        } else if (predicted_class == 2) {  // Buy
            result.direction = 1;
            result.signal_strength = output[2];
        } else {  // Neutral
            result.direction = 0;
            result.signal_strength = 0.0;
        }

        return result;
    }

public:
    LOBPredictor() 
        : predictor_(std::make_unique<ONNXPredictor>()),
          buffer_index_(0), buffer_full_(false) {
        feature_buffer_.resize(SEQUENCE_LENGTH, std::vector<float>(FEATURE_COUNT, 0.0f));
    }

    ~LOBPredictor() = default;

    bool initialize(const std::string& model_path, const std::string& normalization_path) {
        // Initialize ONNX predictor
        if (!predictor_->initialize(model_path)) {
            return false;
        }

        // Load normalization parameters
        try {
            std::ifstream file(normalization_path);
            if (!file.is_open()) {
                std::cerr << "Cannot open normalization file: " << normalization_path << std::endl;
                return false;
            }

            file >> normalization_params_;
            std::cout << "Loaded normalization parameters from: " << normalization_path << std::endl;

            return true;

        } catch (const std::exception& e) {
            std::cerr << "Failed to load normalization parameters: " << e.what() << std::endl;
            return false;
        }
    }

    PredictionResult predict(const OrderBook& order_book) {
        if (!predictor_->isInitialized()) {
            return PredictionResult{};
        }

        try {
            // Extract features from order book
            auto features = extractFeatures(order_book);

            // Normalize features
            auto normalized_features = normalizeFeatures(features);

            // Update feature buffer
            updateFeatureBuffer(normalized_features);

            // Get sequence input
            auto sequence_input = getSequenceInput();

            // Make prediction
            auto output = predictor_->predict(sequence_input);

            // Interpret prediction
            return interpretPrediction(output);

        } catch (const std::exception& e) {
            std::cerr << "LOB prediction error: " << e.what() << std::endl;
            return PredictionResult{};
        }
    }

private:
    std::mutex buffer_mutex_;
};

class IVSurfacePredictor {
private:
    std::unique_ptr<ONNXPredictor> predictor_;

public:
    IVSurfacePredictor() 
        : predictor_(std::make_unique<ONNXPredictor>()) {
    }

    ~IVSurfacePredictor() = default;

    bool initialize(const std::string& model_path) {
        return predictor_->initialize(model_path);
    }

    IVPrediction predictIV(double moneyness, double time_to_expiry, 
                            double vix_level, const std::vector<double>& market_features) {
        IVPrediction result;
        result.timestamp = std::chrono::high_resolution_clock::now();

        if (!predictor_->isInitialized()) {
            return result;
        }

        try {
            auto input = prepareIVInput(moneyness, time_to_expiry, vix_level, market_features);
            auto output = predictor_->predict(input);

            if (!output.empty()) {
                result.implied_volatility = output[0];
                result.confidence = 0.8;  // Simplified confidence
            }

            return result;

        } catch (const std::exception& e) {
            std::cerr << "IV prediction error: " << e.what() << std::endl;
            return result;
        }
    }

private:
    std::vector<float> prepareIVInput(double moneyness, double time_to_expiry,
                                       double vix_level, const std::vector<double>& market_features) {
        std::vector<float> input;
        input.push_back(static_cast<float>(moneyness));
        input.push_back(static_cast<float>(time_to_expiry));
        input.push_back(static_cast<float>(vix_level));

        // Add market features (pad or truncate to expected size)
        for (size_t i = 0; i < 7 && i < market_features.size(); ++i) {
            input.push_back(static_cast<float>(market_features[i]));
        }

        // Pad if necessary
        while (input.size() < 10) {
            input.push_back(0.0f);
        }

        return input;
    }
};

class EnsemblePredictor {
private:
    std::unique_ptr<LOBPredictor> lob_predictor_;
    std::unique_ptr<IVSurfacePredictor> iv_predictor_;
    std::unique_ptr<ONNXPredictor> ensemble_predictor_;
    double lob_weight_;
    double iv_weight_;
    std::vector<PredictionResult> recent_predictions_;
    std::mutex predictions_mutex_;

    PredictionResult combineSimple(const PredictionResult& lob_pred, 
                                   const IVPrediction& iv_pred) {
        PredictionResult result;
        result.timestamp = std::chrono::high_resolution_clock::now();

        // Simple weighted combination
        result.signal_strength = lob_weight_ * lob_pred.signal_strength;
        result.confidence = lob_weight_ * lob_pred.confidence + iv_weight_ * iv_pred.confidence;
        result.direction = lob_pred.direction;  // Use LOB direction for now

        // Combine probabilities if available
        if (lob_pred.probabilities.size() == 3) {
            result.probabilities = lob_pred.probabilities;
        }

        return result;
    }

    PredictionResult combineWithEnsemble(const PredictionResult& lob_pred,
                                         const IVPrediction& iv_pred) {
        try {
            auto input = prepareCombinedInput(lob_pred, iv_pred);
            auto output = ensemble_predictor_->predict(input);

            PredictionResult result;
            result.timestamp = std::chrono::high_resolution_clock::now();

            if (output.size() >= 3) {
                result.probabilities = {output[0], output[1], output[2]};

                auto max_it = std::max_element(output.begin(), output.end());
                int predicted_class = static_cast<int>(std::distance(output.begin(), max_it));

                result.confidence = *max_it;

                if (predicted_class == 0) {
                    result.direction = -1;
                    result.signal_strength = -output[0];
                } else if (predicted_class == 2) {
                    result.direction = 1;
                    result.signal_strength = output[2];
                } else {
                    result.direction = 0;
                    result.signal_strength = 0.0;
                }
            }

            return result;

        } catch (const std::exception& e) {
            std::cerr << "Ensemble combination error: " << e.what() << std::endl;
            return combineSimple(lob_pred, iv_pred);
        }
    }

    std::vector<float> prepareCombinedInput(const PredictionResult& lob_pred,
                                              const IVPrediction& iv_pred) {
        std::vector<float> input;

        // LOB prediction features
        if (lob_pred.probabilities.size() == 3) {
            for (double prob : lob_pred.probabilities) {
                input.push_back(static_cast<float>(prob));
            }
        } else {
            input.push_back(static_cast<float>(lob_pred.signal_strength));
            input.push_back(static_cast<float>(lob_pred.confidence));
            input.push_back(0.0f);
        }

        // IV prediction features
        input.push_back(static_cast<float>(iv_pred.implied_volatility));
        input.push_back(static_cast<float>(iv_pred.confidence));
        input.push_back(0.0f);

        // Additional features (pad to expected input size)
        while (input.size() < 9) {
            input.push_back(0.0f);
        }

        return input;
    }

public:
    EnsemblePredictor() 
        : lob_predictor_(std::make_unique<LOBPredictor>()),
          iv_predictor_(std::make_unique<IVSurfacePredictor>()),
          ensemble_predictor_(std::make_unique<ONNXPredictor>()),
          lob_weight_(0.6), iv_weight_(0.4) {
    }

    ~EnsemblePredictor() = default;

    bool initialize(const std::string& lob_model_path, 
                    const std::string& iv_model_path,
                    const std::string& ensemble_model_path) {

        bool lob_ok = lob_predictor_->initialize(lob_model_path, "models/lob_normalization_params.json");
        bool iv_ok = iv_predictor_->initialize(iv_model_path);
        bool ensemble_ok = ensemble_predictor_->initialize(ensemble_model_path);

        std::cout << "Ensemble initialization - LOB: " << (lob_ok ? "OK" : "FAIL") 
                  << ", IV: " << (iv_ok ? "OK" : "FAIL")
                  << ", Ensemble: " << (ensemble_ok ? "OK" : "FAIL") << std::endl;

        return lob_ok || iv_ok;  // At least one model should work
    }

    PredictionResult predictCombined(const OrderBook& order_book, 
                                     const std::vector<double>& market_features) {
        try {
            // Get LOB prediction
            auto lob_pred = lob_predictor_->predict(order_book);

            // Get IV prediction (simplified - using dummy values for now)
            double moneyness = 0.0;  // log(strike/spot) - would need actual options data
            double time_to_expiry = 0.25;  // 3 months
            double vix_level = 20.0;  // Assume VIX level

            auto iv_pred = iv_predictor_->predictIV(moneyness, time_to_expiry, vix_level, market_features);

            // Store recent predictions
            {
                std::lock_guard<std::mutex> lock(predictions_mutex_);
                recent_predictions_.push_back(lob_pred);
                if (recent_predictions_.size() > 1000) {
                    recent_predictions_.erase(recent_predictions_.begin());
                }
            }

            // Combine predictions
            if (ensemble_predictor_->isInitialized()) {
                return combineWithEnsemble(lob_pred, iv_pred);
            } else {
                return combineSimple(lob_pred, iv_pred);
            }

        } catch (const std::exception& e) {
            std::cerr << "Ensemble prediction error: " << e.what() << std::endl;
            return PredictionResult{};
        }
    }

    std::vector<PredictionResult> getRecentPredictions(size_t count) const {
        std::lock_guard<std::mutex> lock(predictions_mutex_);

        if (recent_predictions_.size() <= count) {
            return recent_predictions_;
        }

        return std::vector<PredictionResult>(
            recent_predictions_.end() - count, 
            recent_predictions_.end()
        );
    }

    void setWeights(double lob_weight, double iv_weight) {
        lob_weight_ = lob_weight;
        iv_weight_ = iv_weight;
    }
};

} // namespace quantx
