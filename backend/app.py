from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime, timedelta
from utils import generate_features_for_inference, validate_company_code, get_company_list, generate_real_features_for_inference
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Global variables for model artifacts
model = None
scaler_X = None
scaler_y = None
le = None
feature_names = None

def load_model_artifacts():
    """Load all model artifacts with proper error handling"""
    global model, scaler_X, scaler_y, le, feature_names
    
    try:
        # Load model
        model_path = './model/best_model_xgboost.pkl'
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info("✅ Model loaded successfully")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load X scaler (non-critical)
        scaler_X_path = './model/scaler_X.pkl'
        if os.path.exists(scaler_X_path):
            try:
                scaler_X = joblib.load(scaler_X_path)
                logger.info(f"✅ X Scaler loaded successfully - fitted: {hasattr(scaler_X, 'scale_')}")
                
                # Check if scaler is properly fitted
                if not hasattr(scaler_X, 'scale_') or scaler_X.scale_ is None:
                    logger.warning("⚠️ X Scaler is not properly fitted - will proceed without scaling")
                    scaler_X = None
            except Exception as e:
                logger.warning(f"⚠️ Error loading X Scaler: {str(e)} - will proceed without scaling")
                scaler_X = None
        else:
            logger.warning("⚠️ X Scaler file not found - will proceed without scaling")
            scaler_X = None
        
        # Load y scaler (non-critical) - NOTE: Will not be used for predictions
        scaler_y_path = './model/scaler_y.pkl'
        if os.path.exists(scaler_y_path):
            try:
                scaler_y = joblib.load(scaler_y_path)
                logger.info(f"✅ Y Scaler loaded successfully - fitted: {hasattr(scaler_y, 'scale_')} (NOTE: Will not be applied to predictions)")
                
                # Check if scaler is properly fitted
                if not hasattr(scaler_y, 'scale_') or scaler_y.scale_ is None:
                    logger.warning("⚠️ Y Scaler is not properly fitted")
                    scaler_y = None
            except Exception as e:
                logger.warning(f"⚠️ Error loading Y Scaler: {str(e)}")
                scaler_y = None
        else:
            logger.warning("⚠️ Y Scaler file not found")
            scaler_y = None
        
        # Load label encoder
        le_path = './model/label_encoder_code.pkl'
        if os.path.exists(le_path):
            le = joblib.load(le_path)
            logger.info("✅ Label encoder loaded successfully")
        else:
            raise FileNotFoundError(f"Label encoder file not found: {le_path}")
        
        # Load feature names
        features_path = './model/features.json'
        if os.path.exists(features_path):
            with open(features_path) as f:
                feature_names = json.load(f)
            logger.info(f"✅ Feature names loaded successfully ({len(feature_names)} features)")
        else:
            raise FileNotFoundError(f"Features file not found: {features_path}")
        
        logger.info("🎉 Model artifacts loaded successfully")
        if scaler_X is None:
            logger.warning("⚠️ No X scaler available - predictions will use raw feature values")
        logger.info("ℹ️ Y scaling will NOT be applied to predictions - raw model output will be returned")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading model artifacts: {str(e)}")
        raise

def validate_scalers():
    """Validate that scalers are properly fitted"""
    issues = []
    
    if scaler_X is not None:
        if not hasattr(scaler_X, 'scale_') or scaler_X.scale_ is None:
            issues.append("X Scaler is not fitted")
        else:
            logger.info(f"X Scaler info - n_features: {scaler_X.n_features_in_}, scale shape: {scaler_X.scale_.shape}")
    
    if scaler_y is not None:
        if not hasattr(scaler_y, 'scale_') or scaler_y.scale_ is None:
            issues.append("Y Scaler is not fitted")
        else:
            logger.info(f"Y Scaler info - n_features: {scaler_y.n_features_in_}, scale shape: {scaler_y.scale_.shape} (NOTE: Will not be used)")
    
    return issues

# Load artifacts on startup
try:
    load_model_artifacts()
    validation_issues = validate_scalers()
    if validation_issues:
        logger.warning(f"Scaler validation issues: {validation_issues}")
        logger.warning("App will continue without problematic scalers")
except Exception as e:
    logger.error(f"Failed to initialize application: {str(e)}")
    raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with detailed status"""
    status_info = {
        'status': 'healthy' if all([model, le, feature_names]) else 'unhealthy',
        'timestamp': datetime.now().isoformat(),
        'components': {
            'model_loaded': model is not None,
            'scaler_X_loaded': scaler_X is not None,
            'scaler_X_fitted': scaler_X is not None and hasattr(scaler_X, 'scale_') and scaler_X.scale_ is not None,
            'scaler_y_loaded': scaler_y is not None,
            'scaler_y_fitted': scaler_y is not None and hasattr(scaler_y, 'scale_') and scaler_y.scale_ is not None,
            'label_encoder_loaded': le is not None,
            'features_loaded': feature_names is not None,
            'feature_count': len(feature_names) if feature_names else 0,
            'y_scaling_applied': False  # Indicates Y scaling is disabled
        }
    }
    
    return jsonify(status_info)

@app.route('/companies', methods=['GET'])
def get_companies():
    """Get list of available company codes"""
    try:
        if le is None:
            return jsonify({'error': 'Label encoder not loaded'}), 500
            
        companies = get_company_list(le)
        return jsonify({
            'companies': companies,
            'total_count': len(companies)
        })
    except Exception as e:
        logger.error(f"Error fetching companies: {str(e)}")
        return jsonify({'error': 'Failed to fetch company list'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint with improved error handling"""
    try:
        # Check if all required components are loaded
        if not all([model, le, feature_names]):
            return jsonify({'error': 'Model components not properly loaded'}), 500
        
        data = request.get_json()
        
        # Validate input data
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        if 'code' not in data or 'date' not in data:
            return jsonify({'error': 'Both "code" and "date" are required'}), 400

        company_code = data['code'].upper().strip()
        date_str = data['date']

        # Validate company code
        if not validate_company_code(company_code, le):
            return jsonify({
                'error': f'Company code "{company_code}" not found in training data'
            }), 400

        # Validate date format
        try:
            input_date = pd.to_datetime(date_str)
            # Check if date is not too far in the future (optional constraint)
            max_date = datetime.now() + timedelta(days=365)
            if input_date > pd.to_datetime(max_date):
                return jsonify({
                    'error': 'Date cannot be more than 1 year in the future'
                }), 400
        except Exception:
            return jsonify({
                'error': 'Invalid date format. Use YYYY-MM-DD'
            }), 400

        # Encode company name
        try:
            encoded_code = le.transform([company_code])[0]
        except Exception as e:
            return jsonify({
                'error': f'Error encoding company code: {str(e)}'
            }), 400

        # Generate features for this input
        feature_df = generate_features_for_inference(date_str, encoded_code)
        
        if feature_df is None or feature_df.empty:
            return jsonify({
                'error': 'Failed to generate features for prediction'
            }), 500

        # Ensure column order and fill missing values
        missing_features = set(feature_names) - set(feature_df.columns)
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            for feature in missing_features:
                feature_df[feature] = 0

        X_input = feature_df[feature_names].fillna(0)
        logger.info(f"Input features shape: {X_input.shape}")

        # Apply scaling only if scaler is available and fitted
        if scaler_X is not None and hasattr(scaler_X, 'scale_') and scaler_X.scale_ is not None:
            try:
                X_scaled = scaler_X.transform(X_input)
                logger.info("✅ Features scaled successfully")
            except Exception as e:
                logger.error(f"❌ Error scaling features: {str(e)}")
                return jsonify({
                    'error': f'Feature scaling failed: {str(e)}'
                }), 500
        else:
            logger.warning("⚠️ Using unscaled features (scaler not available)")  
            X_scaled = X_input.values

        # Make prediction
        try:
            y_pred_raw = model.predict(X_scaled)
            
            # Use raw prediction without any Y scaling
            if len(y_pred_raw.shape) == 1:
                y_pred = y_pred_raw[0]
            else:
                y_pred = y_pred_raw[0][0] if y_pred_raw.shape[1] == 1 else y_pred_raw[0]
            
            logger.info("✅ Using raw model prediction (no Y scaling applied)")
                
        except Exception as e:
            logger.error(f"❌ Model prediction failed: {str(e)}")
            return jsonify({
                'error': f'Model prediction failed: {str(e)}'
            }), 500

        # Get prediction confidence (if available)
        confidence = None
        if hasattr(model, 'predict_proba'):
            try:
                # For classification models
                proba = model.predict_proba(X_scaled)
                confidence = float(np.max(proba))
            except:
                pass

        response = {
            'predicted_price': round(float(y_pred), 2),
            'company': company_code,
            'date': date_str,
            'timestamp': datetime.now().isoformat(),
            'confidence': confidence,
            'scaling_info': {
                'x_scaled': scaler_X is not None,
                'y_scaled': False  # Always False now
            }
        }

        logger.info(f"✅ Prediction successful for {company_code} on {date_str}: ${y_pred:.2f} (raw model output)")
        return jsonify(response)

    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint for multiple companies/dates"""
    try:
        if not all([model, le, feature_names]):
            return jsonify({'error': 'Model components not properly loaded'}), 500
            
        data = request.get_json()
        
        if not data or 'predictions' not in data:
            return jsonify({'error': 'Invalid input format'}), 400
        
        predictions = data['predictions']
        results = []
        
        for idx, pred_data in enumerate(predictions):
            try:
                if 'code' not in pred_data or 'date' not in pred_data:
                    results.append({
                        'error': f'Missing code or date in prediction {idx}',
                        'index': idx
                    })
                    continue
                
                # Use the same prediction logic as single prediction
                company_code = pred_data['code'].upper().strip()
                date_str = pred_data['date']
                
                if not validate_company_code(company_code, le):
                    results.append({
                        'error': f'Invalid company code: {company_code}',
                        'index': idx
                    })
                    continue
                
                encoded_code = le.transform([company_code])[0]
                feature_df = generate_features_for_inference(date_str, encoded_code)
                
                X_input = feature_df[feature_names].fillna(0)
                
                # Apply scaling if available
                if scaler_X is not None and hasattr(scaler_X, 'scale_'):
                    X_scaled = scaler_X.transform(X_input)
                else:
                    X_scaled = X_input.values
                
                y_pred_raw = model.predict(X_scaled)
                
                # Use raw prediction without any Y scaling
                if len(y_pred_raw.shape) == 1:
                    y_pred = y_pred_raw[0]
                else:
                    y_pred = y_pred_raw[0][0] if y_pred_raw.shape[1] == 1 else y_pred_raw[0]
                
                results.append({
                    'predicted_price': round(float(y_pred), 2),
                    'company': company_code,
                    'date': date_str,
                    'index': idx
                })
                
            except Exception as e:
                results.append({
                    'error': str(e),
                    'index': idx
                })
        
        return jsonify({
            'results': results,
            'total_predictions': len(predictions),
            'successful_predictions': len([r for r in results if 'predicted_price' in r])
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/debug/scalers', methods=['GET'])
def debug_scalers():
    """Debug endpoint to check scaler status"""
    debug_info = {
        'scaler_X': {
            'loaded': scaler_X is not None,
            'fitted': scaler_X is not None and hasattr(scaler_X, 'scale_') and scaler_X.scale_ is not None,
            'n_features': scaler_X.n_features_in_ if scaler_X and hasattr(scaler_X, 'n_features_in_') else None,
            'scale_shape': scaler_X.scale_.shape if scaler_X and hasattr(scaler_X, 'scale_') and scaler_X.scale_ is not None else None
        },
        'scaler_y': {
            'loaded': scaler_y is not None,
            'fitted': scaler_y is not None and hasattr(scaler_y, 'scale_') and scaler_y.scale_ is not None,
            'n_features': scaler_y.n_features_in_ if scaler_y and hasattr(scaler_y, 'n_features_in_') else None,
            'scale_shape': scaler_y.scale_.shape if scaler_y and hasattr(scaler_y, 'scale_') and scaler_y.scale_ is not None else None,
            'note': 'Y scaler is loaded but NOT applied to predictions'
        },
        'feature_names_count': len(feature_names) if feature_names else 0,
        'y_scaling_applied': False
    }
    
    return jsonify(debug_info)

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('DEBUG', 'True').lower() == 'true'
    
    logger.info(f"Starting Flask app on port {port} with debug={debug_mode}")
    app.run(host='0.0.0.0', port=port, debug=debug_mode)