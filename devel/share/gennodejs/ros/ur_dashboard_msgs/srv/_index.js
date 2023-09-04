
"use strict";

let IsProgramSaved = require('./IsProgramSaved.js')
let Load = require('./Load.js')
let GetLoadedProgram = require('./GetLoadedProgram.js')
let RawRequest = require('./RawRequest.js')
let GetSafetyMode = require('./GetSafetyMode.js')
let GetProgramState = require('./GetProgramState.js')
let IsInRemoteControl = require('./IsInRemoteControl.js')
let Popup = require('./Popup.js')
let AddToLog = require('./AddToLog.js')
let IsProgramRunning = require('./IsProgramRunning.js')
let GetRobotMode = require('./GetRobotMode.js')

module.exports = {
  IsProgramSaved: IsProgramSaved,
  Load: Load,
  GetLoadedProgram: GetLoadedProgram,
  RawRequest: RawRequest,
  GetSafetyMode: GetSafetyMode,
  GetProgramState: GetProgramState,
  IsInRemoteControl: IsInRemoteControl,
  Popup: Popup,
  AddToLog: AddToLog,
  IsProgramRunning: IsProgramRunning,
  GetRobotMode: GetRobotMode,
};
