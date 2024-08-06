
$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();
    $('.progress').hide();
    $('.w3-container').hide();
    $('.wait_for_loading').hide();
    $('#btn-predict-multiple').hide();
    // Upload Preview

    function readURL(input, index, waste_class) {
        if (input[index]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                img_number = "#img_number_" + index
                gallery_number = "gallery_"+index

                $('.image-section').append("<div id = " + gallery_number + ">")
                $("#"+gallery_number).css({
                    'margin': '5px',
                    'border': '1px solid #ccc',
                    'float': 'left',
                    'width': '152px',
                    'height': '158px',
                })
                $("#"+gallery_number).append("<div id=\"img_number_" + index + "\" class=\"imgs\"></div>")
                $(img_number).css('background-image', 'url(' + e.target.result + ')');
                $(img_number).css('text-align', 'center');

                $("#"+gallery_number).append("<div class=\"desc\">"+waste_class+"</div>")
                $('.desc').css({
                    "padding":"4px",
                })

                $('.img-preview').append("</div>")
            }
            reader.readAsDataURL(input[index]);
        }
    };

    //Called if images got selected
    $('#multi_class_upload').change(function () {
        $('#btn-predict-multiple').show();


        $('.container_bar').remove()
        $('#result').text('');
        $('#result').hide();
    })

    let images_to_send = 0;
    let imgs = [];
    function split_upload() {
        images_to_send = document.getElementById('multi_class_upload').files.length;
        if (images_to_send == 0) {
            $('#msg').html('<span style="color:red">Select at least one file</span>');
            $('#msg').show()
            return;
        }
        if (images_to_send > 10) {
            $('#msg').html('<span style="color:red">Please don\'t select more than 35 images! </span>');
            $('#msg').show()
            return;
        }
        imgs = document.getElementById('multi_class_upload').files
        while(images_to_send > 0){
            var form_data = new FormData();
            form_data.append("files[]", imgs[images_to_send-1])
            form_data.append("index", images_to_send-1)
            $.ajax({
                type: 'POST',
                url: '/classificate_multiple_predict',
                dataType: 'json', // what to expect back from server
                cache: false,
                contentType: false,
                processData: false,
                data: form_data,
                async: true,
                success: function (data) {

                    var classes_array = ['Glas', 'Organic', 'Paper', 'Restmuell', 'Wertstoff']
                    var prob_array = JSON.parse(data["prob"]).map(function (x) {
                        return parseFloat(x)
                    });

                    prob = Math.max(...prob_array)
                    index = prob_array.indexOf(prob)
                    waste_class = classes_array[index]

                    readURL(imgs, data["indexPredicted"], waste_class)
                }
            })
            images_to_send--;
        }


    }

    //uploading multiple (botton clicked)
    $('#btn-predict-multiple').click(function(){
        $(this).hide()
        $('.upload-label').hide()
        $('.loader-multiple').show();
        split_upload()


        images_to_send = 0;
        received_acks = 0;
        $('.image-section').show();
    })

});
