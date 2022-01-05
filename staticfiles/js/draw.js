canvas = document.querySelector("#canvas")

// get canvas 2D context and set him correct size
var ctx = canvas.getContext('2d');
ctx.canvas.width = 300;
ctx.canvas.height = 300;
// last known position
var pos = { x: 0, y: 0 };

document.addEventListener('mousemove', draw);
document.addEventListener('mousedown', setPosition);
document.addEventListener('mouseenter', setPosition);


// new position from mouse event
function setPosition(e) {
    rect = canvas.getBoundingClientRect();
    x = rect.x;
    y = rect.y;
    pos.x = e.clientX - x;
    pos.y = e.clientY - y;
}

function draw(e) {
  // mouse left button must be pressed
  if (e.buttons !== 1) return;

  ctx.beginPath(); // begin

  ctx.lineWidth = 5;
  ctx.lineCap = 'round';
  ctx.strokeStyle = 'black';

  ctx.moveTo(pos.x, pos.y); // from
  setPosition(e);
  ctx.lineTo(pos.x, pos.y); // to

  ctx.stroke(); // draw it!
}

function createWhiteBackground() {
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
}

function saveImage() {
    createWhiteBackground()
    dataURI = canvas.toDataURL()
    postImageData(dataURI)
}

function postImageData(uri) {
    $.post("", {
        img:uri,
        csrfmiddlewaretoken: {{ csrf_token }},
    });
}