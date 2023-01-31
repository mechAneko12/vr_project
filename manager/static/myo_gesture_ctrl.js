/*Bluetooth Function--------------------------------------*/

var myoController;
var myobluetoothDevice;
let accelerometerData, gyroscopeData, emgData0, emgData1, orientationData;

var emg_arr = new Array();
let sample_num = 32;
var predicted_class = 0;

var acc_arr_x = new Array();
var calib_flag = false;
var reset_flag = false;
//myo deviceをconnect
document.querySelector('#startNotifications').addEventListener('click', function(event) {
  event.stopPropagation();
  event.preventDefault();
  console.log("Begin BLE scanning");
  myoController = new MyoWebBluetooth("Sugita");
  myoController.connect();
  myobluetoothDevice = true;
  if(myobluetoothDevice){
    console.log("connecting...");
  }
  

  myoController.onStateChange(function(state){

    emgData0 = state.emgData0;
    emgData1 = state.emgData1;
    orientationData = state.orientation;
    accelerometerData = state.accelerometer;
    gyroscopeData = state.gyroscope;
    //console.log(emgData0);
    //console.log(emgData1);
    //console.log(orientationData);
    //console.log(accelerometerData);
    //console.log(gyroscopeData);
    if (emgData0 != undefined && emgData1 != undefined){
      emg_arr.push(emgData0);
      emg_arr.push(emgData1);
    }
    

    if(emg_arr.length == (32 + sample_num)){
      response = post_emg_arr(emg_arr.flat());
      response.then(function(data){
        predicted_class = data.predicted_class;
        console.log(predicted_class);
      });

      emg_arr.splice(0, (32 + sample_num));
    }

    //calibration and reset by gesture
    if (accelerometerData != undefined){
      acc_arr_x.push(accelerometerData.x);
    }
    if(acc_arr_x.length == 50){
      var large_count = acc_arr_x.filter(x => Math.abs(x) > 0.7).length;
      var small_count = acc_arr_x.filter(x => Math.abs(x) < 0.2).length;
      if(large_count > 40){
        reset_flag = true;
      }else{
        reset_flag = false;
      }

      if(small_count > 40){
        calib_flag = true;
      }else{
        calib_flag = false;
      }

      acc_arr_x.splice(0, 1);
    }

  });
});

//myo deviceをdisconnect
document.querySelector('#disconnect').addEventListener('click', function(event) {
  console.log("disconnect");
  event.stopPropagation();
  event.preventDefault();
  if (!myobluetoothDevice) {
    return;
  }
  myoController.disconnect();
  document.blue.src = "https://res.cloudinary.com/hx3z2s9d0/image/upload/v1577098188/neko_3.gif";
});

/*get cookie*/

function getCookie(name) {
  var cookieValue = null;
  if (document.cookie && document.cookie !== '') {
      var cookies = document.cookie.split(';');
      for (var i = 0; i < cookies.length; i++) {
          var cookie = jQuery.trim(cookies[i]);
          // Does this cookie string begin with the name we want?
          if (cookie.substring(0, name.length + 1) === (name + '=')) {
              cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
              break;
          }
      }
  }
  return cookieValue;
}
var csrftoken = getCookie('csrftoken');

function csrfSafeMethod(method) {
    // these HTTP methods do not require CSRF protection
    return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
}
$.ajaxSetup({
    beforeSend: function(xhr, settings) {
        if (!csrfSafeMethod(settings.type) && !this.crossDomain) {
            xhr.setRequestHeader("X-CSRFToken", csrftoken);
        }
    }
});


/*ajax function--------------------------------------*/
function post_emg_arr(emg_arr){
  return $.ajax({
      url: '/return_class/',
      type: 'POST',
      headers:{'X-CSRFToken': '{{csrf_token}}'},
      
      dataType: 'json',
      //contentType: 'application/json',
      data: {
        'emg_arr': emg_arr
      }
  }).done(function(data, textStatus, jqXHR) {
      return data;
  });
}





/*.jslib Function---------------------------------------*/
function classdata(){
  return predicted_class;
}

function quaterniondata(){
  if(orientationData == undefined){
    return Array(0.0,0.0,0.0, 1.0).join();
  }else{
    return Array(orientationData.x, orientationData.y, orientationData.z, orientationData.w).join();
  }
}

function flags(){
  var flag_arr = new Array(0,0);
  if(calib_flag == true){
    flag_arr[0] = 1;
  }
  if(reset_flag == true){
    flag_arr[1] = 1;
  }
  return flag_arr.join();
}


/*Myo Armband*/


const services = {
  controlService: {
    name: 'control service',
    uuid: 'd5060001-a904-deb9-4748-2c7f4a124842'
  },
  imuDataService :{
    name: 'IMU Data Service',
    uuid: 'd5060002-a904-deb9-4748-2c7f4a124842'
  },
  emgDataService: {
    name: 'EMG Data Service',
    uuid: 'd5060005-a904-deb9-4748-2c7f4a124842'
  },
  batteryService: {
    name: 'battery service',
    uuid: 0x180f
  },
  classifierService: {
    name: 'classifier service',
    uuid: 'd5060003-a904-deb9-4748-2c7f4a124842'
  }
}

const characteristics = {
  commandCharacteristic: {
    name: 'command characteristic',
    uuid: 'd5060401-a904-deb9-4748-2c7f4a124842'
  },
  imuDataCharacteristic: {
    name: 'imu data characteristic',
    uuid: 'd5060402-a904-deb9-4748-2c7f4a124842'
  },
  batteryLevelCharacteristic: {
    name: 'battery level characteristic',
    uuid: 0x2a19
  },
  classifierEventCharacteristic: {
    name: 'classifier event characteristic',
    uuid: 'd5060103-a904-deb9-4748-2c7f4a124842'
  },
  emgData0Characteristic: {
    name: 'EMG Data 0 characteristic',
    uuid: 'd5060105-a904-deb9-4748-2c7f4a124842'
  },
  emgData1Characteristic: {
    name: 'EMG Data 1 characteristic',
    uuid: 'd5060205-a904-deb9-4748-2c7f4a124842'
  },
  emgData2Characteristic: {
    name: 'EMG Data 2 characteristic',
    uuid: 'd5060305-a904-deb9-4748-2c7f4a124842'
  },
  emgData3Characteristic: {
    name: 'EMG Data 3 characteristic',
    uuid: 'd5060405-a904-deb9-4748-2c7f4a124842'
  }
}

//var unityInstance;

var _this;
var state = {};
var previousPose;
var MyobluetoothDevice;

class MyoWebBluetooth{
  constructor(name){
    _this = this;
    this.name = name;
    this.services = services;
    this.characteristics = characteristics;

    this.standardServer;
  }

  connect(){
    return navigator.bluetooth.requestDevice({
      acceptAllDevices:true,
      /*filters: [
        {name: this.name},
        {
          services: [services.batteryService.uuid,
                     services.imuDataService.uuid,
                     services.controlService.uuid,
                     services.emgDataService.uuid]
        }
      ],*/
      optionalServices: [
        services.imuDataService.uuid,
        services.controlService.uuid,
        services.emgDataService.uuid]
    })
    .then(device => {
      console.log('Device discovered', device.name);
      MyobluetoothDevice = device;
      return device.gatt.connect();
    })
    .then(server => {
      console.log('server device: '+ Object.keys(server.device));
      document.blue.src = "https://res.cloudinary.com/hx3z2s9d0/image/upload/v1577098188/neko_2.gif";
      //unityInstance = UnityLoader.instantiate("unityContainer", "Build/%UNITY_WEB_NAME%.json", {onProgress: UnityProgress});
      this.getServices([services.controlService, services.emgDataService, services.imuDataService], [characteristics.commandCharacteristic, characteristics.emgData0Characteristic, characteristics.emgData1Characteristic, characteristics.emgData2Characteristic, characteristics.emgData3Characteristic, characteristics.imuDataCharacteristic], server);
    })
    .catch(error => {console.log('error',error)})
  }

  disconnect(){
    MyobluetoothDevice.gatt.disconnect();
  }

  getServices(requestedServices, requestedCharacteristics, server){
    this.standardServer = server;

    requestedServices.filter((service) => {
      if(service.uuid == services.controlService.uuid){
        _this.getControlService(requestedServices, requestedCharacteristics, this.standardServer);
      }
    })
  }

  getControlService(requestedServices, requestedCharacteristics, server){
      let controlService = requestedServices.filter((service) => { return service.uuid == services.controlService.uuid});
      let commandChar = requestedCharacteristics.filter((char) => {return char.uuid == characteristics.commandCharacteristic.uuid});

      // Before having access to IMU, EMG and Pose data, we need to indicate to the Myo that we want to receive this data.
      return server.getPrimaryService(controlService[0].uuid)
      .then(service => {
        console.log('getting service: ', controlService[0].name);
        return service.getCharacteristic(commandChar[0].uuid);
      })
      .then(characteristic => {
        console.log('getting characteristic: ', commandChar[0].name);
        // return new Buffer([0x01,3,emg_mode,imu_mode,classifier_mode]);
        // The values passed in the buffer indicate that we want to receive all data without restriction;
        let commandValue = new Uint8Array([0x01,3,0x02,0x03,0x01]);
        characteristic.writeValue(commandValue);
      })
      .then(_ => {
        let IMUService = requestedServices.filter((service) => {return service.uuid == services.imuDataService.uuid});
        let EMGService = requestedServices.filter((service) => {return service.uuid == services.emgDataService.uuid});
        //let classifierService = requestedServices.filter((service) => {return service.uuid == services.classifierService.uuid});

        let IMUDataChar = requestedCharacteristics.filter((char) => {return char.uuid == characteristics.imuDataCharacteristic.uuid});
        let EMGDataChar0 = requestedCharacteristics.filter((char) => {return char.uuid == characteristics.emgData0Characteristic.uuid});
        let EMGDataChar1 = requestedCharacteristics.filter((char) => {return char.uuid == characteristics.emgData1Characteristic.uuid});
        let EMGDataChar2 = requestedCharacteristics.filter((char) => {return char.uuid == characteristics.emgData2Characteristic.uuid});
        let EMGDataChar3 = requestedCharacteristics.filter((char) => {return char.uuid == characteristics.emgData3Characteristic.uuid});
        //let classifierEventChar = requestedCharacteristics.filter((char) => {return char.uuid == characteristics.classifierEventCharacteristic.uuid});

        if(IMUService.length > 0){
          console.log('getting service: ', IMUService[0].name);
          _this.getIMUData(IMUService[0], IMUDataChar[0], server);
        }
        if(EMGService.length > 0){
          console.log('getting service: ', EMGService[0].name);
          _this.getEMGData(EMGService[0], EMGDataChar0[0], server);
          _this.getEMGData(EMGService[0], EMGDataChar1[0], server);
          _this.getEMGData(EMGService[0], EMGDataChar2[0], server);
          _this.getEMGData(EMGService[0], EMGDataChar3[0], server);
        }
      })
      .catch(error =>{
        console.log('error: ', error);
      })
  }

  handleIMUDataChanged(event){
    //byteLength of ImuData DataView object is 20.
    // imuData return {{orientation: {w: *, x: *, y: *, z: *}, accelerometer: Array, gyroscope: Array}}
    let imuData = event.target.value;

    let orientationW = event.target.value.getInt16(0, true) / 16384;
    let orientationX = event.target.value.getInt16(2, true) / 16384;
    let orientationY = event.target.value.getInt16(4, true) / 16384;
    let orientationZ = event.target.value.getInt16(6, true) / 16384;

    let accelerometerX = event.target.value.getInt16(8, true) / 2048;
    let accelerometerY = event.target.value.getInt16(10, true) / 2048;
    let accelerometerZ = event.target.value.getInt16(12, true) / 2048;

    let gyroscopeX = event.target.value.getInt16(14, true) / 16;
    let gyroscopeY = event.target.value.getInt16(16, true) / 16;
    let gyroscopeZ = event.target.value.getInt16(18, true) / 16;

    var data = {
      orientation: {
        x: orientationX,
        y: orientationY,
        z: orientationZ,
        w: orientationW
      },
      accelerometer: {
        x: accelerometerX,
        y: accelerometerY,
        z: accelerometerZ
      },
      gyroscope: {
        x: gyroscopeX,
        y: gyroscopeY,
        z: gyroscopeZ
      }
    }

    state = {
      orientation: data.orientation,
      accelerometer: data.accelerometer,
      gyroscope: data.gyroscope
    }

    _this.onStateChangeCallback(state);
  }

  onStateChangeCallback() {}

  getIMUData(service, characteristic, server){
    return server.getPrimaryService(service.uuid)
    .then(newService => {
      console.log('getting characteristic: ', characteristic.name);
      return newService.getCharacteristic(characteristic.uuid)
    })
    .then(char => {
      char.startNotifications().then(res => {
        char.addEventListener('characteristicvaluechanged', _this.handleIMUDataChanged);
      })
    })
  }

  getEMGData(service, characteristic, server){
    return server.getPrimaryService(service.uuid)
    .then(newService => {
      console.log('getting characteristic: ', characteristic.name);
      return newService.getCharacteristic(characteristic.uuid)
    })
    .then(char => {
      console.log("A");
      char.startNotifications().then(res => {
        char.addEventListener('characteristicvaluechanged', _this.handleEMGDataChanged);
      })
    })
  }
 

  handleEMGDataChanged(event){
      //byteLength of ImuData DataView object is 20.
      // imuData return {{orientation: {w: *, x: *, y: *, z: *}, accelerometer: Array, gyroscope: Array}}
      let emgData = event.target.value;

      let sample1 = [
        emgData.getInt8(0),
        emgData.getInt8(1),
        emgData.getInt8(2),
        emgData.getInt8(3),
        emgData.getInt8(4),
        emgData.getInt8(5),
        emgData.getInt8(6),
        emgData.getInt8(7)
      ]

      let sample2 = [
        emgData.getInt8(8),
        emgData.getInt8(9),
        emgData.getInt8(10),
        emgData.getInt8(11),
        emgData.getInt8(12),
        emgData.getInt8(13),
        emgData.getInt8(14),
        emgData.getInt8(15)
      ]

      state.emgData0 = sample1;
      state.emgData1 = sample2;

      _this.onStateChangeCallback(state);
  }

  onStateChange(callback){
    _this.onStateChangeCallback = callback;
  }
}