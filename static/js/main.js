// Main JavaScript functions for the application

// Handle MLFlow synchronization
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips - usando la modalità nativa di bootstrap 5
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    if (tooltipTriggerList.length > 0 && typeof bootstrap !== 'undefined') {
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }

    // MLFlow sync button
    const syncMlflowBtn = document.getElementById('sync-mlflow-btn');
    if (syncMlflowBtn) {
        syncMlflowBtn.addEventListener('click', function() {
            const jobId = this.getAttribute('data-job-id');
            syncMlflowMetrics(jobId);
        });
    }
});

// Function to sync MLFlow metrics
function syncMlflowMetrics(jobId) {
    // Mostra un indicatore di caricamento sul pulsante
    const syncBtn = document.getElementById('sync-mlflow-btn');
    const originalText = syncBtn.innerHTML;
    syncBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Sincronizzazione...';
    syncBtn.disabled = true;

    // Effettua la richiesta API
    fetch(`/api/job/${jobId}/sync_mlflow`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Mostra un messaggio di successo
            toastr.success(data.message);

            // Aggiorna la pagina dopo 2 secondi per mostrare le metriche aggiornate
            setTimeout(() => {
                window.location.reload();
            }, 2000);
        } else {
            // Mostra un messaggio di errore
            toastr.error(data.error || 'Errore durante la sincronizzazione con MLFlow');
            // Ripristina il pulsante
            syncBtn.innerHTML = originalText;
            syncBtn.disabled = false;
        }
    })
    .catch(error => {
        console.error('Errore durante la sincronizzazione:', error);
        toastr.error('Errore di rete durante la sincronizzazione');
        // Ripristina il pulsante
        syncBtn.innerHTML = originalText;
        syncBtn.disabled = false;
    });
}
function syncMlflowMetrics(jobId) {
    if (!jobId) return;

    // Change button state to loading
    const btn = document.getElementById('sync-mlflow-btn');
    if (btn) {
        const originalHtml = btn.innerHTML;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Sincronizzazione...';
        btn.disabled = true;

        // Make AJAX call to sync metrics
        fetch(`/api/job/${jobId}/sync_mlflow`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert('Metriche e artefatti sincronizzati con successo con MLFlow!');
                // Reload page to show updated data
                window.location.reload();
            } else {
                alert('Errore: ' + (data.error || 'Si è verificato un errore sconosciuto durante la sincronizzazione.'));
                btn.innerHTML = originalHtml;
                btn.disabled = false;
            }
        })
        .catch(error => {
            console.error('Errore di sincronizzazione con MLFlow:', error);
            alert('Errore di connessione al server. Riprova più tardi.');
            btn.innerHTML = originalHtml;
            btn.disabled = false;
        });
    }
}

// Function to show toast notification
function showToast(title, message, type) {
    // Create toast element if it doesn't exist
    if (!document.getElementById('toast-container')) {
        const toastContainer = document.createElement('div');
        toastContainer.id = 'toast-container';
        toastContainer.style.position = 'fixed';
        toastContainer.style.top = '20px';
        toastContainer.style.right = '20px';
        toastContainer.style.zIndex = '1050';
        document.body.appendChild(toastContainer);
    }

    const toastId = 'toast-' + Date.now();
    const toastElement = document.createElement('div');
    toastElement.id = toastId;
    toastElement.className = `toast bg-${type === 'error' ? 'danger' : 'success'} text-white`;
    toastElement.setAttribute('role', 'alert');
    toastElement.setAttribute('aria-live', 'assertive');
    toastElement.setAttribute('aria-atomic', 'true');

    const toastHeader = document.createElement('div');
    toastHeader.className = 'toast-header bg-transparent text-white';
    toastHeader.innerHTML = `
        <strong class="me-auto">${title}</strong>
        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast" aria-label="Close"></button>
    `;

    const toastBody = document.createElement('div');
    toastBody.className = 'toast-body';
    toastBody.textContent = message;

    toastElement.appendChild(toastHeader);
    toastElement.appendChild(toastBody);

    document.getElementById('toast-container').appendChild(toastElement);

    // Initialize and show the toast
    const toast = new bootstrap.Toast(toastElement, { autohide: true, delay: 5000 });
    toast.show();

    // Remove the toast after it's hidden
    toastElement.addEventListener('hidden.bs.toast', function() {
        toastElement.remove();
    });
}

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips and popovers
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // Handle model type selection
    const modelTypeSelectors = document.querySelectorAll('.model-type-selector');
    if (modelTypeSelectors) {
        modelTypeSelectors.forEach(button => {
            button.addEventListener('click', function() {
                const modelType = this.getAttribute('data-model-type');
                window.location.href = `/configure?model_type=${modelType}`;
            });
        });
    }

    // Handle model variant selection
    const variantSelectors = document.querySelectorAll('.variant-selector');
    if (variantSelectors) {
        variantSelectors.forEach(card => {
            card.addEventListener('click', function() {
                // Remove selected class from all cards
                variantSelectors.forEach(c => c.classList.remove('border-primary'));

                // Add selected class to clicked card
                this.classList.add('border-primary');

                // Get variant and update form value
                const variant = this.getAttribute('data-variant');
                document.getElementById('model_variant').value = variant;

                // Get default values for this variant
                const defaultValues = JSON.parse(this.getAttribute('data-defaults'));

                // Update form fields with default values
                if (defaultValues) {
                    if (defaultValues.default_epochs) {
                        document.getElementById('epochs').value = defaultValues.default_epochs;
                    }

                    if (defaultValues.default_batch_size) {
                        document.getElementById('batch_size').value = defaultValues.default_batch_size;
                    }

                    if (defaultValues.default_img_size) {
                        document.getElementById('img_size').value = defaultValues.default_img_size;
                    }

                    if (defaultValues.default_learning_rate) {
                        document.getElementById('learning_rate').value = defaultValues.default_learning_rate;
                    }
                }
            });
        });
    }

    // Handle cancel job button
    const cancelButtons = document.querySelectorAll('.cancel-job-btn');
    if (cancelButtons) {
        cancelButtons.forEach(button => {
            button.addEventListener('click', function() {
                const jobId = this.getAttribute('data-job-id');

                if (confirm('Are you sure you want to cancel this training job?')) {
                    // Send request to cancel job
                    fetch(`/api/job/${jobId}/cancel`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        }
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            alert('Job cancelled successfully');
                            window.location.reload();
                        } else {
                            alert(`Error: ${data.error}`);
                        }
                    })
                    .catch(error => {
                        console.error('Error cancelling job:', error);
                        alert('An error occurred while cancelling the job');
                    });
                }
            });
        });
    }
});
function syncMLFlow(jobId) {
    // Show loading indicator
    showToast('Syncing with MLFlow...', 'info');

    fetch(`/api/sync_mlflow/${jobId}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showToast('MLFlow sync completed successfully');
                // Reload the data to show updated metrics
                if (window.location.pathname.includes('/results/')) {
                    loadJobDetails();
                } else if (window.location.pathname.includes('/jobs')) {
                    // Refresh jobs list if on jobs page
                    refreshJobsList();
                }
            } else {
                showToast('MLFlow sync failed: ' + data.error, 'error');
            }
        })
        .catch(err => {
            console.error('Error syncing with MLFlow:', err);
            showToast('Error syncing with MLFlow', 'error');
        });
}