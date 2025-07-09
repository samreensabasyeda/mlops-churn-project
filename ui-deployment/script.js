// Configuration
const API_BASE_URL = window.location.hostname === 'localhost' 
    ? 'http://localhost:8000' 
    : 'http://NODE_IP:30080'; // Replace NODE_IP with actual node IP or use LoadBalancer URL

// Global variables
let apiOnline = false;

// DOM elements
const predictionForm = document.getElementById('predictionForm');
const resultsContent = document.getElementById('resultsContent');
const apiStatus = document.getElementById('apiStatus');
const modelInfo = document.getElementById('modelInfo');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    checkApiStatus();
    loadModelInfo();
    setupFormSubmission();
    
    // Check API status every 30 seconds
    setInterval(checkApiStatus, 30000);
});

// Check API health status
async function checkApiStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            }
        });

        if (response.ok) {
            const data = await response.json();
            apiOnline = true;
            updateApiStatus('online', `✅ API Online - ${data.status}`);
        } else {
            throw new Error('API not responding');
        }
    } catch (error) {
        apiOnline = false;
        updateApiStatus('offline', '❌ API Offline');
        console.error('API health check failed:', error);
    }
}

// Update API status indicator
function updateApiStatus(status, message) {
    apiStatus.className = `api-status ${status}`;
    apiStatus.innerHTML = message;
}

// Load model information
async function loadModelInfo() {
    try {
        const response = await fetch(`${API_BASE_URL}/model-info`);
        
        if (response.ok) {
            const data = await response.json();
            displayModelInfo(data);
        } else {
            throw new Error('Failed to load model info');
        }
    } catch (error) {
        modelInfo.innerHTML = `
            <div class="text-center text-muted">
                <i class="fas fa-exclamation-triangle"></i>
                <p class="mt-2">Could not load model information</p>
            </div>
        `;
        console.error('Model info load failed:', error);
    }
}

// Display model information
function displayModelInfo(data) {
    modelInfo.innerHTML = `
        <div class="model-info-item">
            <span class="model-info-label">Model Status:</span>
            <span class="model-info-value">
                <i class="fas fa-${data.model_loaded ? 'check-circle text-success' : 'times-circle text-danger'}"></i>
                ${data.model_loaded ? 'Loaded' : 'Not Loaded'}
            </span>
        </div>
        <div class="model-info-item">
            <span class="model-info-label">Version:</span>
            <span class="model-info-value">${data.model_version || 'Unknown'}</span>
        </div>
        <div class="model-info-item">
            <span class="model-info-label">Last Updated:</span>
            <span class="model-info-value">${data.last_updated || 'Unknown'}</span>
        </div>
        <div class="model-info-item">
            <span class="model-info-label">Registry Group:</span>
            <span class="model-info-value">${data.registry_group}</span>
        </div>
        <div class="text-center mt-3">
            <button class="btn btn-sm btn-outline-primary" onclick="reloadModel()">
                <i class="fas fa-sync-alt"></i> Reload Model
            </button>
        </div>
    `;
}

// Setup form submission
function setupFormSubmission() {
    predictionForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        if (!apiOnline) {
            showError('API is offline. Please try again later.');
            return;
        }

        // Show loading state
        showLoading();
        
        // Collect form data
        const formData = new FormData(predictionForm);
        const customerData = {};
        
        for (let [key, value] of formData.entries()) {
            // Convert numeric fields
            if (['SeniorCitizen', 'tenure', 'MonthlyCharges'].includes(key)) {
                customerData[key] = key === 'MonthlyCharges' ? parseFloat(value) : parseInt(value);
            } else {
                customerData[key] = value;
            }
        }
        
        try {
            // Make prediction request
            const response = await fetch(`${API_BASE_URL}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(customerData)
            });

            if (response.ok) {
                const result = await response.json();
                displayPredictionResult(result);
            } else {
                const errorData = await response.json();
                showError(errorData.detail || 'Prediction failed');
            }
        } catch (error) {
            showError('Network error. Please check your connection.');
            console.error('Prediction error:', error);
        }
    });
}

// Show loading state
function showLoading() {
    resultsContent.innerHTML = `
        <div class="text-center py-4">
            <div class="spinner-border text-primary mb-3" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="text-muted">Analyzing customer data...</p>
        </div>
    `;
}

// Display prediction results
function displayPredictionResult(result) {
    const riskClass = result.risk_level.toLowerCase();
    const churnText = result.churn_prediction === 1 ? 'LIKELY TO CHURN' : 'LIKELY TO STAY';
    const probability = (result.churn_probability * 100).toFixed(1);
    
    resultsContent.innerHTML = `
        <div class="prediction-result fade-in">
            <div class="risk-indicator ${riskClass} pulse">
                ${probability}%
            </div>
            <h4 class="mb-3">${churnText}</h4>
            <div class="probability-text">
                <strong>Churn Probability:</strong> ${probability}%
            </div>
            <div class="risk-level">
                <span class="badge badge-${riskClass} p-2">
                    ${result.risk_level} Risk
                </span>
            </div>
            <div class="model-version">
                Model: v${result.model_version}
            </div>
            <div class="prediction-timestamp mt-2">
                <small class="text-muted">
                    <i class="fas fa-clock"></i>
                    ${result.prediction_timestamp}
                </small>
            </div>
            
            <div class="recommendations mt-4">
                <h5>💡 Recommendations:</h5>
                <div class="recommendation-list">
                    ${getRecommendations(result)}
                </div>
            </div>
        </div>
    `;
}

// Generate recommendations based on risk level
function getRecommendations(result) {
    const risk = result.risk_level.toLowerCase();
    
    const recommendations = {
        high: [
            '🚨 Immediate retention call recommended',
            '💰 Consider offering discount or promotion',
            '📞 Schedule customer service follow-up',
            '🎁 Provide loyalty rewards or upgrades'
        ],
        medium: [
            '📧 Send targeted retention email campaign',
            '📊 Monitor account activity closely',
            '🤝 Offer customer satisfaction survey',
            '📱 Promote additional services'
        ],
        low: [
            '✅ Customer appears satisfied',
            '📈 Good candidate for upselling',
            '🌟 Consider loyalty program enrollment',
            '📝 Maintain regular communication'
        ]
    };
    
    return recommendations[risk].map(rec => `<div class="recommendation-item">${rec}</div>`).join('');
}

// Show error message
function showError(message) {
    resultsContent.innerHTML = `
        <div class="text-center py-4">
            <i class="fas fa-exclamation-triangle fa-3x text-warning mb-3"></i>
            <h5 class="text-danger">Prediction Failed</h5>
            <p class="text-muted">${message}</p>
            <button class="btn btn-outline-primary btn-sm mt-2" onclick="checkApiStatus()">
                <i class="fas fa-sync-alt"></i> Retry
            </button>
        </div>
    `;
}

// Reload model
async function reloadModel() {
    try {
        const response = await fetch(`${API_BASE_URL}/reload-model`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });

        if (response.ok) {
            const result = await response.json();
            showNotification('success', 'Model reloaded successfully!');
            loadModelInfo(); // Refresh model info
        } else {
            const errorData = await response.json();
            showNotification('error', errorData.detail || 'Model reload failed');
        }
    } catch (error) {
        showNotification('error', 'Network error during model reload');
        console.error('Model reload error:', error);
    }
}

// Show notification
function showNotification(type, message) {
    const notification = document.createElement('div');
    notification.className = `alert alert-${type === 'success' ? 'success' : 'danger'} position-fixed`;
    notification.style.cssText = `
        top: 20px;
        right: 20px;
        z-index: 9999;
        min-width: 300px;
        border-radius: 10px;
    `;
    notification.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : 'exclamation-circle'}"></i>
        ${message}
        <button type="button" class="btn-close" onclick="this.parentElement.remove()"></button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 5000);
}

// Add sample data for testing
function fillSampleData() {
    const sampleData = {
        gender: 'Female',
        SeniorCitizen: '0',
        Partner: 'Yes',
        Dependents: 'No',
        tenure: '12',
        PhoneService: 'Yes',
        MultipleLines: 'No',
        InternetService: 'DSL',
        OnlineSecurity: 'Yes',
        OnlineBackup: 'No',
        DeviceProtection: 'No',
        TechSupport: 'No',
        StreamingTV: 'No',
        StreamingMovies: 'No',
        Contract: 'Month-to-month',
        PaperlessBilling: 'Yes',
        PaymentMethod: 'Electronic check',
        MonthlyCharges: '50.00',
        TotalCharges: '600.00'
    };
    
    Object.keys(sampleData).forEach(key => {
        const element = document.querySelector(`[name="${key}"]`);
        if (element) {
            element.value = sampleData[key];
        }
    });
    
    showNotification('success', 'Sample data filled!');
}

// Add sample data button (for testing)
document.addEventListener('DOMContentLoaded', function() {
    const sampleButton = document.createElement('button');
    sampleButton.type = 'button';
    sampleButton.className = 'btn btn-outline-secondary btn-sm me-2';
    sampleButton.innerHTML = '<i class="fas fa-file-import"></i> Fill Sample Data';
    sampleButton.onclick = fillSampleData;
    
    const predictButton = document.querySelector('.btn-predict');
    predictButton.parentNode.insertBefore(sampleButton, predictButton);
}); 