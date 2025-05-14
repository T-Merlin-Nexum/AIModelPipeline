/**
 * Upload.js for handling dataset uploads
 */

document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('dataset_zip');
    const uploadBtn = document.getElementById('upload-btn');
    const cancelBtn = document.getElementById('cancel-upload');
    const progressContainer = document.getElementById('progress-container');
    const progressBar = document.getElementById('upload-progress');
    const uploadStatusText = document.getElementById('upload-status');
    
    if (!uploadForm || !fileInput) {
        return;
    }
    
    // Hide progress bar initially
    if (progressContainer) {
        progressContainer.style.display = 'none';
    }
    
    // Update file input label with selected filename
    fileInput.addEventListener('change', function() {
        const fileLabel = document.querySelector('.custom-file-label');
        
        if (this.files.length > 0) {
            const fileName = this.files[0].name;
            if (fileLabel) {
                fileLabel.textContent = fileName;
            }
            
            // Check file size
            const fileSize = this.files[0].size;
            const maxSize = 510 * 1024 * 1024; // 500 MB with a small buffer
            
            if (fileSize > maxSize) {
                alert(`File is too large! Maximum size is 500 MB. Your file is ${(fileSize / (1024 * 1024)).toFixed(2)} MB.`);
                this.value = ''; // Clear file input
                if (fileLabel) {
                    fileLabel.textContent = 'Choose dataset ZIP file...';
                }
                return;
            }
            
            // Check file extension
            if (!fileName.toLowerCase().endsWith('.zip')) {
                alert('Only ZIP files are supported!');
                this.value = ''; // Clear file input
                if (fileLabel) {
                    fileLabel.textContent = 'Choose dataset ZIP file...';
                }
                return;
            }
        } else {
            if (fileLabel) {
                fileLabel.textContent = 'Choose dataset ZIP file...';
            }
        }
    });
    
    // Handle form submission with progress tracking
    uploadForm.addEventListener('submit', function(e) {
        // Form validation will be handled by the browser due to the required attributes
        // But we can add extra validation here if needed
        
        if (fileInput.files.length === 0) {
            alert('Please select a ZIP file to upload');
            e.preventDefault();
            return;
        }
        
        // Show progress container
        if (progressContainer) {
            progressContainer.style.display = 'block';
        }
        
        // Disable submit button to prevent multiple submissions
        if (uploadBtn) {
            uploadBtn.disabled = true;
            uploadBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Uploading...';
        }
        
        // For this implementation, we'll let the form submit normally
        // In a production app, you might want to use AJAX to track upload progress
        // This would require server-side support for progress tracking
        if (uploadStatusText) {
            uploadStatusText.textContent = 'Uploading dataset... This may take a while for large files.';
        }
    });
    
    // Handle cancel button
    if (cancelBtn) {
        cancelBtn.addEventListener('click', function() {
            if (confirm('Are you sure you want to cancel the upload?')) {
                window.location.href = '/datasets';
            }
        });
    }
    
    // Inizializzazione dei formati di dataset
    function initializeFormatSelection() {
        // Aggiungi event listener ai radio buttons
        const radioButtons = document.querySelectorAll('input[name="format_type"]');
        radioButtons.forEach(radio => {
            radio.addEventListener('change', function() {
                // Rimuovi la classe border-primary da tutte le cards
                document.querySelectorAll('.format-card').forEach(card => {
                    card.classList.remove('border-primary');
                });
                
                // Aggiungi la classe border-primary alla card selezionata
                const formatValue = this.value;
                const selectedCard = document.querySelector(`.format-card[data-format="${formatValue}"]`);
                if (selectedCard) {
                    selectedCard.classList.add('border-primary');
                }
                
                console.log("Format selected via radio change:", formatValue);
            });
        });
        
        // Aggiungi event listener alle card per renderle cliccabili
        const formatCards = document.querySelectorAll('.format-card');
        formatCards.forEach(card => {
            card.addEventListener('click', function() {
                const formatValue = this.getAttribute('data-format');
                const radioToSelect = document.querySelector(`input[name="format_type"][value="${formatValue}"]`);
                
                if (radioToSelect) {
                    // Seleziona il radio button
                    radioToSelect.checked = true;
                    
                    // Trigger manuale dell'evento change
                    const event = new Event('change', { bubbles: true });
                    radioToSelect.dispatchEvent(event);
                    
                    console.log("Format selected via card click:", formatValue);
                }
            });
        });
    }
    
    // Chiamiamo la funzione di inizializzazione quando il DOM è caricato
    document.addEventListener('DOMContentLoaded', function() {
        initializeFormatSelection();
    });

    // Funzione selectFormat
    window.selectFormat = function(formatValue) {
        // Log available radio buttons for debugging
        console.log("Available radio buttons:", Array.from(document.querySelectorAll('input[name="format_type"]')).map(r => r.value));
        
        // Trova il radio button usando sia l'ID che un selettore più flessibile
        const radio = document.querySelector(`input[name="format_type"][value="${formatValue}"]`);
        
        if (radio) {
            // Check the radio
            radio.checked = true;
            
            // Rimuovi la classe border-primary da tutte le cards
            document.querySelectorAll('.format-card').forEach(card => {
                card.classList.remove('border-primary');
            });
            
            // Aggiungi la classe border-primary alla card selezionata
            const selectedCard = document.querySelector(`.format-card[data-format="${formatValue}"]`);
            if (selectedCard) {
                selectedCard.classList.add('border-primary');
            }
            
            console.log("Format selected:", formatValue);
        } else {
            console.error("Radio button not found for format:", formatValue);
        }
    };

    // Handle format radio button changes
    const formatRadios = document.querySelectorAll('input[name="format_type"]');
    if (formatRadios.length > 0) {
        formatRadios.forEach(radio => {
            radio.addEventListener('change', function() {
                selectFormat(this.value);
            });
        });
    }
    
    // Make entire card clickable
    document.querySelectorAll('.format-card').forEach(card => {
        card.addEventListener('click', function() {
            const format = this.getAttribute('data-format');
            selectFormat(format);
        });
    });
});
