
/**
 * Configure.js per gestire la selezione del modello
 */

document.addEventListener('DOMContentLoaded', function() {
    // Gestione della selezione del tipo di modello (YOLO o RF-DETR)
    const modelTypeSelectors = document.querySelectorAll('.model-type-selector');
    
    modelTypeSelectors.forEach(selector => {
        selector.addEventListener('click', function() {
            const modelType = this.getAttribute('data-model-type');
            
            // Rimuovi la classe selezionata da tutti
            modelTypeSelectors.forEach(s => s.classList.remove('border-primary', 'selected-card'));
            
            // Aggiungi la classe al selezionato
            this.classList.add('border-primary', 'selected-card');
            
            // Reindirizza alla pagina di configurazione con il tipo di modello
            window.location.href = `/configure?model_type=${modelType}`;
        });
    });
    
    // Per le varianti del modello sulla pagina di configurazione
    const variantSelectors = document.querySelectorAll('.variant-selector');
    
    if (variantSelectors.length > 0) {
        // Funzione per selezionare una variante
        window.selectVariant = function(element) {
            // Rimuovi la selezione da tutte le varianti
            variantSelectors.forEach(card => {
                card.classList.remove('border-primary');
            });
            
            // Aggiungi la selezione a quella cliccata
            element.classList.add('border-primary');
            
            // Imposta il valore nel campo nascosto
            const variantValue = element.getAttribute('data-variant');
            document.getElementById('model_variant').value = variantValue;
            
            // Carica i valori predefiniti se disponibili
            const defaults = JSON.parse(element.getAttribute('data-defaults') || '{}');
            
            // Aggiorna i campi del form con i valori predefiniti
            if (defaults.default_epochs) {
                document.getElementById('epochs').value = defaults.default_epochs;
            }
            
            if (defaults.default_batch_size) {
                document.getElementById('batch_size').value = defaults.default_batch_size;
            }
            
            if (defaults.default_img_size) {
                document.getElementById('img_size').value = defaults.default_img_size;
            }
            
            if (defaults.default_learning_rate) {
                document.getElementById('learning_rate').value = defaults.default_learning_rate;
            }
        }
        
        // Aggiungi l'event listener per ogni card
        variantSelectors.forEach(card => {
            card.addEventListener('click', function() {
                selectVariant(this);
            });
        });
        
        // Seleziona la prima variante all'inizio
        selectVariant(variantSelectors[0]);
    }
});
