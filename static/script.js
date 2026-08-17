document.addEventListener("DOMContentLoaded", () => {

const fileInput = document.getElementById("image");
const preview = document.getElementById("preview");

if(fileInput){

fileInput.addEventListener("change", function(e){

const file = e.target.files[0];

if(file){

preview.src = URL.createObjectURL(file);

preview.style.display="block";

}

});

}

const form=document.querySelector("form");

if(form){

form.addEventListener("submit",()=>{

const btn=document.querySelector("button");

btn.innerHTML="⏳ Predicting...";

btn.disabled=true;

});

}

});