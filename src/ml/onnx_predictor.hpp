#pragma once

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>
#include "../core/market_data_handler.hpp"

namespace quantx {

struct PredictionResult {
    double signal_strength;  // -1.0 to 1.0 (sell to buy)
    double confidence;       // 0.0 to 1.0
    int direction;          // -1 (sell), 0 (neutral), 1 (buy)
    std::vector<double> probabilities; // Class probabilities
    std::chrono::high_resolution_clock::time_point timestamp;
    
    PredictionResult() : signal_strength(0.0), confidence(0.0), direction(0) {}
};

struct IVPrediction {
    double implied_volatility;
    double confidence;
    std::chrono::high_resolution_clock::time_point timestamp;
    
    IVPrediction() : implied_volatility(0.0), confidence(0.0) {}
};

class ONNXPredictor {
private:
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::Env> env_;
    Ort::MemoryInfo memory_info_;
    
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;
    
    std::string model_path_;
    bool initialized_;
    
    mutable std::mutex prediction_mutex_;
    
public:
    ONNXPredictor();
    ~ONNXPredictor();
    
    bool initialize(const std::string& model_path);
    bool isInitialized() const { return initialized_; }
    
    std::vector<float> predict(const std::vector<float>& input);
    
    // Model info
    const std::vector<std::string>& getInputNames() const { return input_names_; }
    const std::vector<std::string>& getOutputNames() const { return output_names_; }
    const std::vector<std::vector<int64_t>>& getInputShapes() const { return input_shapes_; }
    const std::vector<std::vector<int64_t>>& getOutputShapes() const { return output_shapes_; }
    
private:
    void extractModelInfo();
};

class LOBPredictor {
private:
    std::unique_ptr<ONNXPredictor> predictor_;
    nlohmann::json normalization_params_;
    
    // Feature extraction parameters
    static constexpr int SEQUENCE_LENGTH = 10;
    static constexpr int FEATURE_COUNT = 14;
    
    // Circular buffer for maintaining sequence
    std::vector<std::vector<float>> feature_buffer_;
    size_t buffer_index_;
    bool buffer_full_;
    
    mutable std::mutex buffer_mutex_;
    
public:
    LOBPredictor();
    ~LOBPredictor();
    
    bool initialize(const std::string& model_path, const std::string& normalization_path);
    
    PredictionResult predict(const OrderBook& order_book);
    
private:
    std::vector<float> extractFeatures(const OrderBook& order_book);
    std::vector<float> normalizeFeatures(const std::vector<float>& features);
    void updateFeatureBuffer(const std::vector<float>& features);
    std::vector<float> getSequenceInput();
    PredictionResult interpretPrediction(const std::vector<float>& output);
};

class IVSurfacePredictor {
private:
    std::unique_ptr<ONNXPredictor> predictor_;
    
public:
    IVSurfacePredictor();
    ~IVSurfacePredictor();
    
    bool initialize(const std::string& model_path);
    
    IVPrediction predictIV(double moneyness, double time_to_expiry, 
                          double vix_level, const std::vector<double>& market_features);
    
private:
    std::vector<float> prepareIVInput(double moneyness, double time_to_expiry,
                                     double vix_level, const std::vector<double>& market_features);
};

class EnsemblePredictor {
private:
    std::unique_ptr<LOBPredictor> lob_predictor_;
    std::unique_ptr<IVSurfacePredictor> iv_predictor_;
    std::unique_ptr<ONNXPredictor> ensemble_predictor_;
    
    // Weights for combining predictions
    double lob_weight_;
    double iv_weight_;
    
    // Recent predictions for ensemble input
    std::vector<PredictionResult> recent_predictions_;
    std::mutex predictions_mutex_;
    
public:
    EnsemblePredictor();
    ~EnsemblePredictor();
    
    bool initialize(const std::string& lob_model_path, 
                   const std::string& iv_model_path,
                   const std::string& ensemble_model_path);
    
    PredictionResult predictCombined(const OrderBook& order_book, 
                                   const std::vector<double>& market_features);
    
    std::vector<PredictionResult> getRecentPredictions(size_t count) const;
    
    void setWeights(double lob_weight, double iv_weight);
    
private:
    PredictionResult combineSimple(const PredictionResult& lob_pred, 
                                  const IVPrediction& iv_pred);
    
    PredictionResult combineWithEnsemble(const PredictionResult& lob_pred,
                                        const IVPrediction& iv_pred);
    
    std::vector<float> prepareCombinedInput(const PredictionResult& lob_pred,
                                           const IVPrediction& iv_pred);
};

} // namespace quantx
