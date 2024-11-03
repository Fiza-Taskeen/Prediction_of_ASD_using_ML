// script.js

function predict() {
    var age = $('#age').val();
    var qchat_score = $('#qchat_score').val();
    var sex = $('#sex').val();
    var ethnicity = $('#ethnicity').val();
    var jaundice = $('#jaundice').val();
    var family_asd = $('#family_asd').val();
    var who_completed = $('#who_completed').val();
    
    // Make AJAX request to backend
    $.ajax({
        type: 'POST',
        url: '/predict',
        contentType: 'application/json',
        data: JSON.stringify({
            AGE_MONS: parseFloat(age),
            QCHAT_10_SCORE: parseInt(qchat_score),
            SEX: parseInt(sex),
            ETHNICITY: ethnicity,
            JAUNDICE: parseInt(jaundice),
            FAMILY_MEM_WITH_ASD: parseInt(family_asd),
            WHO_COMPLETED_THE_TEST: parseInt(who_completed) // Corrected key name
        }),
        success: function(response) {
            $('#predictionResult').text('Prediction: ' + response.prediction);
        },
        error: function(xhr, status, error) {
            console.error('Error:', error);
        }
    });
}
