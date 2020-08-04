//
var myoController;
var myobluetoothDevice;
var csv_flag = false;
var timerID;
let accelerometerData, gyroscopeData, emgData0, emgData1, orientationData;
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
    console.log(emgData0);
    console.log(emgData1);
    console.log(orientationData);
    console.log(accelerometerData);
    console.log(gyroscopeData);
    /*
    if(csv_flag == true && emgData0 !== undefined){
      emgData0 = Object.entries(emgData0);
      array_data.push([Date.now() - start_time, emgData0[0][1], emgData0[1][1], emgData0[2][1], emgData0[3][1], emgData0[4][1], emgData0[5][1], emgData0[6][1], emgData0[7][1]]);
    }*/
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
//



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

var unityInstance;

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
      unityInstance = UnityLoader.instantiate("unityContainer", "Build/%UNITY_WEB_NAME%.json", {onProgress: UnityProgress});
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