
$(document).ready(function () {
    // On load hide all elements which are not needed
    $('#text-accept-uplaod').hide();
    $('#radioDiv').hide();
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();
    $('.progress').hide();
    $('#progressBar').hide();
    $('#button_multiple').hide();
    $('.w3-container').hide();
    $('#selected_images').hide();
    $('.wait_for_loading').hide();
    // Upload Preview
    function readURL(input) {
        console.log(input)
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').css('text-align', 'center');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    //Called if an image got selected.
    $('#imageUpload').change(function () {
        $('#msg').hide()
        $('#selected_images').hide();

        $('.container_bar').remove()
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    //-----------------------------------------------------

    //Called if images got selected
    $('#imagesUpload').change(function () {
        $('#button_multiple').show();
        $('#msg').hide()
        $('#selected_images').show();

        var amount_imgs = document.getElementById('imagesUpload').files.length;
        $('#selected_images').html('<span style="color:green">Selected ' + amount_imgs + ' images. </span>');
    })

    let images_to_send = 0;
    let received_acks = 0;

    //for multiple images upload they are splitted and sent to the server
    function split_upload() {
        if($("#radioDiv input[type='radio']:checked").val() == 'no'){
            $('#msg').html('<span style="color:red">You did not allow to upload the images. Check the buttons above and please try again!</span>');
            $('#msg').show()
            return -1
        }
        // Get the amount of images to send
        images_to_send = document.getElementById('imagesUpload').files.length;
        if (images_to_send == 0) {
            $('#msg').html('<span style="color:red">Select at least one file</span>');
            $('#msg').show()
            return;
        }

        //Here its editable to set the amount of images to send
        if (images_to_send > 30) {
            $('#msg').html('<span style="color:red">Please don\'t select more than 30 images! </span>');
            $('#msg').show()
            return;
        }

        imgs = document.getElementById('imagesUpload').files
        // Images get splitted and sent to the server one by one
        while(images_to_send > 0){
            var form_data = new FormData();
            form_data.append("files[]", imgs[images_to_send-1])
            console.log("sending: " + imgs[images_to_send-1])
            $.ajax({
                type: 'POST',
                url: '/images',
                dataType: 'json', // what to expect back from server
                cache: false,
                contentType: false,
                processData: false,
                data: form_data,
                async: true,
                success: function (data) {
                    //if we get success from the server, we increase the received_acks and update the progress bar
                    received_acks++;
                    progress_bar_upload(received_acks, document.getElementById('imagesUpload').files.length)
                }
            })
            images_to_send--;
        }
        images_to_send = 0;
        received_acks = 0;
    }

    //method for updating the progress bar
    function progress_bar_upload(received, amount_to_upload) {
        $('.wait_for_loading').hide();
        if (received == 0 || amount_to_upload == 0) {
            return -1
        }
        if(received == 1){
            $('.w3-container').fadeIn(600);
        }
        var elem = document.getElementById("myBar");
        var width = 100 * (received / amount_to_upload);
        if (width >= 100) {
            elem.style.width = width + '%';
            document.getElementById("myP").innerHTML = "Successfully uploaded " + amount_to_upload + " photos!";
        } else {
            elem.style.width = width + '%';
            var num = width * 1 / amount_to_upload;
            num = num.toFixed(0)
            if ($('#demo').length) {
                document.getElementById("demo").innerHTML = received;
                document.getElementById("end").innerHTML = amount_to_upload;
            }
        }
    }

    //uploading multiple (botton clicked)
    $('#button_multiple').click(function(){
        $('.wait_for_loading').show();
        $('#selected_images').hide();
        $('#msg').hide()
        $(this).hide()
        document.getElementById("myBar").style.width = "0%"
        document.getElementById("myP").innerHTML = "Added <span id=\"demo\">0</span> of <span id=\"end\">&infin;</span> photos";
        split_upload()
    })

    // Predict (botton clicked)
    $('#btn-predict').click(function () {
        $('#msg').hide()
        $('.container_bar').remove()
        var form_data = new FormData($('#upload-file')[0]);
        form_data.append('checkbox', $("#radioDiv input[type='radio']:checked").val())
        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Array for all classes our model can predict
                var classes_array = ['Glas', 'Organic', 'Paper', 'Restmuell', 'Wertstoff']
                // Get the result from the server
                var prob_array = JSON.parse(data["prob"]).map(function (x) {
                    return parseFloat(x)
                });

                // Get and display the result
                $('.loader').hide();
                $('.progress').show();
                for(let i = 0; i < 5; i++){
                    prob = Math.max(...prob_array)
                    index = prob_array.indexOf(prob)
                    waste_class = classes_array[index]
                    prob_array.splice(index, 1)
                    classes_array.splice(index,1)
                    if(100*prob < 5){
                        continue;
                    }
                    else if(100*prob > 70){
                        color = "orange"
                        size = "130%"
                        if(100*prob > 80){
                            color = "blue"
                            size = "160%"
                                if(100*prob > 90){
                                    color = "green"
                                    size = "190%"
                                }
                        }
                    }
                    else if(100*prob < 20){
                        color = "red"
                        size = "90%"
                    }
                    else{
                        color = "black"
                        size = "100%"
                    }

                    hint = ""
                    var hints = [/*Glas:*/'Throw it in the glass container!',
                        /*Organic:*/'Throw it in the organic container (green)!',
                        /*Paper:*/'It belongs to the paper container (blue)!',
                        /*Residual waste:*/'Throw it in the residual waste container (black one)!',
                        /*Recyclable:*/'Throw it in the recyclable container (yellow one)!']
                    if(i == 0){
                        //Class with highest probability
                        //Add hints to the result!
                        hint = ": " + hints[index]
                    }

                    $('.progress').append("<div id=\"bar_container_"+ i + "\" class=\"container_bar\" \"></div>")
                    bar_container = '#bar_container_'+i
                    $(bar_container).append("<p id=\"text_number_"+ i +"\" class=\"text-bar\"></p>")
                    text = '#text_number_'+i
                    $(text).text(""+waste_class + hint)
                    hint = ""
                    $(text).css({
                               'font-size' : size,
                               'color' : color,
                            })

                    $(bar_container).append("<div id=\"bar_number_"+ i +"\" class=\"progress-bar\" role=\"progressbar\"></div>")
                    bar_number = "#bar_number_"+ i
                    $(bar_number).text(Math.round(100*prob,2) + "%")
                    $(bar_number).css("width",""+Math.round(100*prob,2)+"%")
                }
            },
        });
    });

});
