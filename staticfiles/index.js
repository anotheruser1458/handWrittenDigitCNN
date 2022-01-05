// wait for the content of the window element
// to load, then performs the operations.
// This is considered best practice.
window.addEventListener('load', ()=>{

	resize(); // Resizes the canvas once the window loads
	document.addEventListener('mousedown', startPainting);
	document.addEventListener('mouseup', stopPainting);
	document.addEventListener('mousemove', sketch);
	window.addEventListener('resize', resize);
});

const canvas = document.querySelector('#canvas');

// Context for the canvas for 2 dimensional operations
const ctx = canvas.getContext('2d');

// Resizes the canvas to the available size of the window.
function resize(){
ctx.canvas.width = 500;
ctx.canvas.height = 500;
}

// Stores the initial position of the cursor
let coord = {x:0 , y:0};

// This is the flag that we are going to use to
// trigger drawing
let paint = false;

// Updates the coordianates of the cursor when
// an event e is triggered to the coordinates where
// the said event is triggered.
function getPosition(event){
coord.x = event.clientX - canvas.offsetLeft;
coord.y = event.clientY - canvas.offsetTop;
}

// The following functions toggle the flag to start
// and stop drawing
function startPainting(event){
paint = true;
getPosition(event);
}
function stopPainting(){
paint = false;
}

function sketch(event){
if (!paint) return;
ctx.beginPath();

ctx.lineWidth = 30;

// Sets the end of the lines drawn
// to a round shape.
ctx.lineCap = 'round';

ctx.strokeStyle = 'black';

// The cursor to start drawing
// moves to this coordinate
ctx.moveTo(coord.x, coord.y);

// The position of the cursor
// gets updated as we move the
// mouse around.
getPosition(event);

// A line is traced from start
// coordinate to this coordinate
ctx.lineTo(coord.x , coord.y);

// Draws the line.
ctx.stroke();
}

function saveImage() {
  var canvasData = canvas.toDataURL("image/png");
  var xmlHttpReq = false;

  if (window.XMLHttpRequest) {
    ajax = new XMLHttpRequest();
  }
  else if (window.ActiveXObject) {
    ajax = new ActiveXObject("Microsoft.XMLHTTP");
  }

  ajax.open("POST", "http://127.0.0.1:8000/draw", false);
  ajax.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");
  ajax.onreadystatechange = function() {
    console.log(ajax.responseText);
  }
  ajax.send("imgData=" + canvasData);


}


function ajaxPost(){
    var imgData=ctx.getImageData(0,0,canvas.width,canvas.height);
var data=imgData.data;
for(var i=0;i<data.length;i+=4){
    if(data[i+3]<255){
        data[i]=255;
        data[i+1]=255;
        data[i+2]=255;
        data[i+3]=255;
    }
}
ctx.putImageData(imgData,0,0);

    var canvasData = canvas.toDataURL("image/png");

    $.post("", {
        img:canvasData
    }, function(data,status,xhr){
    console.log(data);
    });


}


function test(){
    console.log("hello");
}


