import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class RiskAssessment:
    """
    Risk assessment module for ocean forecasting
    """
    
    def __init__(self):
        # Risk thresholds for different parameters
        self.thresholds = {
            'sea_level': {
                'low': {'min': -1.0, 'max': 1.0},
                'medium': {'min': -2.0, 'max': 2.0},
                'high': {'outside': [-2.0, 2.0]}
            },
            'wind_speed_10m': {
                'low': {'min': 0, 'max': 15},
                'medium': {'min': 15, 'max': 25},
                'high': {'min': 25, 'max': np.inf}
            },
            'pressure_msl': {
                'low': {'min': 1000, 'max': 1030},
                'medium': {'min': 980, 'max': 1000},
                'high': {'outside': [980, 1030]}
            },
            'temperature_2m': {
                'low': {'min': 0, 'max': 35},
                'medium': {'min': -10, 'max': 0},
                'high': {'outside': [-10, 35]}
            }
        }
        
        # Risk messages
        self.risk_messages = {
            'sea_level': {
                'High': "Extreme sea level variations detected. High flood or surge risk.",
                'Medium': "Moderate sea level changes. Monitor coastal conditions.",
                'Low': "Sea level within normal range."
            },
            'wind_speed_10m': {
                'High': "High wind speeds detected. Strong storm conditions possible.",
                'Medium': "Moderate wind speeds. Be aware of changing weather conditions.",
                'Low': "Wind speeds within normal range."
            },
            'pressure_msl': {
                'High': "Extreme atmospheric pressure. Severe weather system approaching.",
                'Medium': "Low atmospheric pressure. Weather system developing.",
                'Low': "Atmospheric pressure within normal range."
            },
            'temperature_2m': {
                'High': "Extreme temperature conditions detected.",
                'Medium': "Temperature outside normal range.",
                'Low': "Temperature within expected range."
            }
        }
    
    def assess_risk(self, current_value, predictions, uncertainties, parameter):
        """
        Assess risk level based on current value, predictions, and uncertainties
        
        Args:
            current_value: Current parameter value
            predictions: Array of predicted values
            uncertainties: Array of prediction uncertainties
            parameter: Parameter name (key in thresholds)
            
        Returns:
            risk_level: 'Low', 'Medium', or 'High'
            risk_message: Descriptive risk message
        """
        if parameter not in self.thresholds:
            return 'Low', 'Unknown parameter for risk assessment'
        
        try:
            # Get parameter thresholds
            param_thresholds = self.thresholds[parameter]
            
            # Calculate risk factors
            current_risk = self._assess_single_value(current_value, param_thresholds)
            prediction_risk = self._assess_predictions(predictions, param_thresholds)
            uncertainty_risk = self._assess_uncertainty(uncertainties, predictions)
            trend_risk = self._assess_trend(predictions)
            
            # Combine risk factors
            risk_scores = {
                'current': current_risk,
                'prediction': prediction_risk,
                'uncertainty': uncertainty_risk,
                'trend': trend_risk
            }
            
            # Calculate overall risk
            overall_risk = self._calculate_overall_risk(risk_scores)
            risk_level = self._risk_score_to_level(overall_risk)
            
            # Get appropriate message
            risk_message = self.risk_messages[parameter][risk_level]
            
            # Add specific details
            detailed_message = self._create_detailed_message(
                risk_level, risk_message, current_value, predictions, uncertainties, parameter
            )
            
            return risk_level, detailed_message
            
        except Exception as e:
            return 'Low', f'Error in risk assessment: {str(e)}'
    
    def _assess_single_value(self, value, thresholds):
        """Assess risk for a single value"""
        if 'high' in thresholds:
            if 'min' in thresholds['high'] and value >= thresholds['high']['min']:
                return 3  # High risk
            if 'max' in thresholds['high'] and value <= thresholds['high']['max']:
                return 3  # High risk
            if 'outside' in thresholds['high']:
                min_val, max_val = thresholds['high']['outside']
                if value < min_val or value > max_val:
                    return 3  # High risk
        
        if 'medium' in thresholds:
            if 'min' in thresholds['medium'] and 'max' in thresholds['medium']:
                if thresholds['medium']['min'] <= value <= thresholds['medium']['max']:
                    return 2  # Medium risk
        
        if 'low' in thresholds:
            if 'min' in thresholds['low'] and 'max' in thresholds['low']:
                if thresholds['low']['min'] <= value <= thresholds['low']['max']:
                    return 1  # Low risk
        
        return 1  # Default to low risk
    
    def _assess_predictions(self, predictions, thresholds):
        """Assess risk based on prediction values"""
        if len(predictions) == 0:
            return 1
        
        risk_scores = []
        for pred in predictions:
            risk_scores.append(self._assess_single_value(pred, thresholds))
        
        # Return maximum risk from predictions
        return max(risk_scores)
    
    def _assess_uncertainty(self, uncertainties, predictions):
        """Assess risk based on prediction uncertainties"""
        if len(uncertainties) == 0 or len(predictions) == 0:
            return 1
        
        # Calculate relative uncertainty
        mean_prediction = np.mean(np.abs(predictions))
        mean_uncertainty = np.mean(uncertainties)
        
        if mean_prediction == 0:
            relative_uncertainty = 1
        else:
            relative_uncertainty = mean_uncertainty / mean_prediction
        
        # Risk thresholds for uncertainty
        if relative_uncertainty > 0.5:  # 50% uncertainty
            return 3  # High risk
        elif relative_uncertainty > 0.2:  # 20% uncertainty
            return 2  # Medium risk
        else:
            return 1  # Low risk
    
    def _assess_trend(self, predictions):
        """Assess risk based on prediction trends"""
        if len(predictions) < 2:
            return 1
        
        # Calculate trend (slope)
        x = np.arange(len(predictions))
        trend = np.polyfit(x, predictions, 1)[0]
        
        # Calculate rate of change
        rate_of_change = np.abs(trend) * len(predictions)
        
        # Risk thresholds for trend
        if rate_of_change > 2.0:  # Rapid change
            return 3  # High risk
        elif rate_of_change > 1.0:  # Moderate change
            return 2  # Medium risk
        else:
            return 1  # Low risk
    
    def _calculate_overall_risk(self, risk_scores):
        """Calculate overall risk score from individual components"""
        # Weighted average of risk components
        weights = {
            'current': 0.3,
            'prediction': 0.4,
            'uncertainty': 0.2,
            'trend': 0.1
        }
        
        overall_score = 0
        for component, score in risk_scores.items():
            overall_score += weights.get(component, 0) * score
        
        return overall_score
    
    def _risk_score_to_level(self, score):
        """Convert numeric risk score to risk level"""
        if score >= 2.5:
            return 'High'
        elif score >= 1.5:
            return 'Medium'
        else:
            return 'Low'
    
    def _create_detailed_message(self, risk_level, base_message, current_value, 
                                predictions, uncertainties, parameter):
        """Create detailed risk message with specific information"""
        details = []
        
        # Add current value
        details.append(f"Current {parameter.replace('_', ' ')}: {current_value:.2f}")
        
        # Add prediction range
        if len(predictions) > 0:
            min_pred = np.min(predictions)
            max_pred = np.max(predictions)
            details.append(f"Predicted range: {min_pred:.2f} to {max_pred:.2f}")
        
        # Add uncertainty information
        if len(uncertainties) > 0:
            avg_uncertainty = np.mean(uncertainties)
            details.append(f"Average uncertainty: ±{avg_uncertainty:.2f}")
        
        # Combine base message with details
        detailed_message = f"{base_message} " + " | ".join(details)
        
        return detailed_message
    
    def assess_multiple_parameters(self, parameter_data, predictions_data, uncertainties_data):
        """
        Assess risk for multiple parameters simultaneously
        
        Args:
            parameter_data: Dict of {parameter: current_value}
            predictions_data: Dict of {parameter: predictions_array}
            uncertainties_data: Dict of {parameter: uncertainties_array}
            
        Returns:
            overall_risk_level: Combined risk level
            risk_summary: Dictionary of individual parameter risks
        """
        parameter_risks = {}
        risk_scores = []
        
        for parameter in parameter_data.keys():
            if parameter in predictions_data and parameter in uncertainties_data:
                risk_level, risk_message = self.assess_risk(
                    parameter_data[parameter],
                    predictions_data[parameter],
                    uncertainties_data[parameter],
                    parameter
                )
                
                parameter_risks[parameter] = {
                    'level': risk_level,
                    'message': risk_message
                }
                
                # Convert risk level to numeric score
                risk_scores.append(self._level_to_score(risk_level))
        
        # Calculate overall risk
        if risk_scores:
            overall_score = max(risk_scores)  # Use maximum risk
            overall_risk_level = self._score_to_level(overall_score)
        else:
            overall_risk_level = 'Low'
        
        return overall_risk_level, parameter_risks
    
    def _level_to_score(self, level):
        """Convert risk level to numeric score"""
        level_map = {'Low': 1, 'Medium': 2, 'High': 3}
        return level_map.get(level, 1)
    
    def _score_to_level(self, score):
        """Convert numeric score to risk level"""
        if score >= 3:
            return 'High'
        elif score >= 2:
            return 'Medium'
        else:
            return 'Low'
    
    def generate_recommendations(self, risk_level, parameter, predictions):
        """
        Generate actionable recommendations based on risk assessment
        
        Args:
            risk_level: Risk level ('Low', 'Medium', 'High')
            parameter: Parameter name
            predictions: Prediction array
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if risk_level == 'High':
            if parameter == 'sea_level':
                recommendations.extend([
                    "🚨 Immediate action required - monitor coastal areas",
                    "📍 Consider evacuation of low-lying areas",
                    "🛡️ Activate flood defense systems",
                    "📞 Alert emergency services and coastal authorities"
                ])
            elif parameter == 'wind_speed_10m':
                recommendations.extend([
                    "🌪️ Secure all loose objects and equipment",
                    "⛵ Advise against marine activities",
                    "🏠 Check building integrity and storm preparations",
                    "📡 Monitor weather updates continuously"
                ])
            elif parameter == 'pressure_msl':
                recommendations.extend([
                    "🌀 Severe weather system approaching",
                    "📊 Monitor all meteorological indicators",
                    "🚢 Advise vessels to seek shelter",
                    "⚠️ Prepare for potential storm surge"
                ])
        
        elif risk_level == 'Medium':
            recommendations.extend([
                "👀 Continue monitoring conditions closely",
                "📈 Review forecast updates regularly",
                "🔄 Update contingency plans",
                "📋 Prepare emergency response protocols"
            ])
        
        else:  # Low risk
            recommendations.extend([
                "✅ Continue routine monitoring",
                "📊 Maintain regular data collection",
                "🔍 Watch for any developing patterns"
            ])
        
        return recommendations
    
    def create_risk_report(self, assessment_results):
        """
        Create a comprehensive risk report
        
        Args:
            assessment_results: Results from risk assessment
            
        Returns:
            Formatted risk report string
        """
        report_lines = []
        report_lines.append("🌊 OCEAN FORECASTING RISK REPORT")
        report_lines.append("=" * 40)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report_lines.append("")
        
        if isinstance(assessment_results, dict):
            # Multiple parameters
            for param, risk_info in assessment_results.items():
                report_lines.append(f"📊 {param.replace('_', ' ').title()}:")
                report_lines.append(f"   Risk Level: {risk_info['level']}")
                report_lines.append(f"   Details: {risk_info['message']}")
                report_lines.append("")
        else:
            # Single parameter
            risk_level, risk_message = assessment_results
            report_lines.append(f"Risk Level: {risk_level}")
            report_lines.append(f"Assessment: {risk_message}")
        
        return "\n".join(report_lines)
