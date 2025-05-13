/**
 * Charts.js for training job metrics visualization
 */

// Function to create a line chart for training metrics
function createMetricsChart(canvasId, data, options = {}) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // Default configuration
    const defaultOptions = {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
            duration: 0 // general animation time
        },
        hover: {
            animationDuration: 0 // duration of animations when hovering an item
        },
        responsiveAnimationDuration: 0, // animation duration after a resize
        elements: {
            line: {
                tension: 0.3 // smoother curves
            },
            point: {
                radius: 2,
                hitRadius: 10,
                hoverRadius: 5
            }
        },
        plugins: {
            legend: {
                position: 'top',
            },
            tooltip: {
                mode: 'index',
                intersect: false,
                backgroundColor: 'rgba(0, 0, 0, 0.8)',
                titleColor: '#ffffff',
                bodyColor: '#ffffff',
                borderColor: 'rgba(0, 0, 0, 0.2)',
                borderWidth: 1
            }
        },
        scales: {
            x: {
                display: true,
                title: {
                    display: true,
                    text: 'Epoch'
                },
                grid: {
                    display: false
                }
            },
            y: {
                display: true,
                title: {
                    display: true,
                    text: 'Value'
                },
                grid: {
                    color: 'rgba(200, 200, 200, 0.1)'
                }
            }
        }
    };
    
    // Merge options
    const chartOptions = { ...defaultOptions, ...options };
    
    // Create and return chart
    return new Chart(ctx, {
        type: 'line',
        data: data,
        options: chartOptions
    });
}

// Function to update chart with new data
function updateMetricsChart(chart, newData) {
    chart.data = newData;
    chart.update();
}

// Function to create a dataset object for Chart.js
function createDataset(label, data, color) {
    return {
        label: label,
        data: data,
        backgroundColor: color,
        borderColor: color,
        fill: false,
        borderWidth: 2
    };
}

// Function to prepare data for metrics chart
function prepareChartData(metrics, epochKey = 'epoch') {
    // Extract epochs
    const epochs = metrics[epochKey] || [];
    
    // Define metrics to show and their colors
    const metricsConfig = {
        'loss': 'rgba(255, 99, 132, 1)',
        'precision': 'rgba(54, 162, 235, 1)',
        'recall': 'rgba(75, 192, 192, 1)',
        'mAP50': 'rgba(153, 102, 255, 1)',
        'mAP50-95': 'rgba(255, 159, 64, 1)'
    };
    
    // Prepare datasets
    const datasets = [];
    
    for (const [metric, color] of Object.entries(metricsConfig)) {
        if (metrics[metric]) {
            datasets.push(createDataset(metric, metrics[metric], color));
        }
    }
    
    return {
        labels: Array.from({length: epochs.length}, (_, i) => i + 1),
        datasets: datasets
    };
}

// Function to fetch metrics for a job and update the chart
function fetchAndUpdateMetrics(jobId, chart) {
    fetch(`/api/jobs/${jobId}/status`)
        .then(response => response.json())
        .then(data => {
            if (data.metrics) {
                // Prepare chart data
                const chartData = prepareChartData(data.metrics);
                
                // Update chart
                updateMetricsChart(chart, chartData);
                
                // Update progress
                const progressBar = document.getElementById('trainingProgress');
                if (progressBar && data.progress) {
                    progressBar.style.width = `${data.progress}%`;
                    progressBar.setAttribute('aria-valuenow', data.progress);
                    
                    // Aggiorna il testo della barra di progresso
                    const progressText = document.getElementById('progressText');
                    if (progressText) {
                        progressText.textContent = `${Math.round(data.progress)}%`;
                    }
                }
                
                // Update status
                const statusBadge = document.getElementById('jobStatus');
                if (statusBadge) {
                    statusBadge.textContent = data.status.toUpperCase();
                    
                    // Reset all classes
                    statusBadge.className = 'badge rounded-pill';
                    
                    // Add color based on status
                    switch (data.status) {
                        case 'pending':
                            statusBadge.classList.add('bg-secondary');
                            break;
                        case 'running':
                            statusBadge.classList.add('bg-primary');
                            break;
                        case 'completed':
                            statusBadge.classList.add('bg-success');
                            break;
                        case 'failed':
                            statusBadge.classList.add('bg-danger');
                            break;
                        case 'cancelled':
                            statusBadge.classList.add('bg-warning');
                            break;
                        default:
                            statusBadge.classList.add('bg-secondary');
                    }
                }
                
                // Se il job è completato, mostra il messaggio e attiva il pulsante
                if (data.status === 'completed') {
                    // Show complete message and enable view results button
                    const completeMsg = document.getElementById('trainingCompleteMessage');
                    if (completeMsg) {
                        completeMsg.classList.remove('d-none');
                    }
                    
                    const resultsBtn = document.getElementById('viewResultsBtn');
                    if (resultsBtn) {
                        resultsBtn.classList.remove('disabled');
                    }
                } else if (data.status === 'failed') {
                    // Show error message
                    const errorMsg = document.getElementById('trainingErrorMessage');
                    if (errorMsg) {
                        errorMsg.classList.remove('d-none');
                        if (data.error_message) {
                            const errorDetails = errorMsg.querySelector('.error-details');
                            if (errorDetails) {
                                errorDetails.textContent = data.error_message;
                            }
                        }
                    }
                }
            }
        })
        .catch(error => {
            console.error('Error fetching metrics:', error);
        });
}

// Initialize monitoring for a job
function initJobMonitoring(jobId) {
    // Create initial empty chart
    const metricsCanvas = document.getElementById('metricsChart');
    if (metricsCanvas) {
        const chart = createMetricsChart('metricsChart', {
            labels: [],
            datasets: []
        });
        
        // Fetch metrics and update
        fetchAndUpdateMetrics(jobId, chart);
        
        // Aggiorna ogni 5 secondi
        const updateInterval = setInterval(() => {
            fetchAndUpdateMetrics(jobId, chart);
            
            // Controlla se il job è completato o fallito per fermare l'aggiornamento
            fetch(`/api/jobs/${jobId}/status`)
                .then(response => response.json())
                .then(data => {
                    // Se lo stato è completed, failed o cancelled, ferma l'aggiornamento
                    if (['completed', 'failed', 'cancelled'].includes(data.status)) {
                        console.log(`Job ${jobId} è ${data.status}, fermando gli aggiornamenti automatici`);
                        clearInterval(updateInterval);
                        
                        // Aggiorna la pagina se lo stato è cambiato
                        const statusBadge = document.getElementById('jobStatus');
                        if (statusBadge && statusBadge.textContent.toLowerCase() !== data.status.toUpperCase()) {
                            location.reload();
                        }
                    }
                })
                .catch(error => console.error('Error fetching job status:', error));
        }, 5000);
    }
}
