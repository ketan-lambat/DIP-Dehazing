

function readURL(input){
  if (true) {
     document.getElementById("output").style.display = "display: none;";
     var reader = new FileReader();
     formdata = new FormData();
     reader.onload = function(e) {
        $('#blah').attr('src', e.target.result).width(150)
                    .height(200);

        formdata.append('image',  $('#blah').attr('src'));
         $.ajax({
            type: "POST",
            url: "/send",
		    data: formdata,
		    cache: false,
		    processData: false,
		    contentType: false
         }).success(function(response){
            console.log("HI");
            console.log(response);
            $('#output').attr('src', response).width(150)
                    .height(200);
          });
     }

     reader.readAsDataURL(input.files[0]);
     document.getElementById("arrow").style.display = "inline-block";
  }


}




