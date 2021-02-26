// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {InferenceHandler, SessionHandler} from '../../backend';
import {ExecutionPlan} from '../../execution-plan';
import {Graph} from '../../graph';
import {Operator} from '../../operators';
import {OpSet, resolveOperator} from '../../opset';
import {Session} from '../../session';
import {Tensor} from '../../tensor';
import {ProtoUtil} from '../../util';
import {getInstance, InferenceContext} from '../../wasm-binding-core';
import {WasmBackend} from '../backend-wasm';

// import {CPU_OP_RESOLVE_RULES} from '../cpu/op-resolve-rules';

import {WasmInferenceHandler} from './inference-handler';
import {OP_INFO_RESOLVE_RULES} from './op-resolve-rules-vnext';
import {OperatorInfo} from './op-vnext';

// import {WASM_OP_RESOLVE_RULES} from './op-resolve-rules';

export class WasmSessionHandler implements SessionHandler {
  customExecutionPlan: ExecutionPlan;
  wasmContext: InferenceContext;
  private bindingInstance: ReturnType<typeof getInstance>;

  // private opResolveRules: ReadonlyArray<OpSet.ResolveRule>;
  constructor(readonly backend: WasmBackend, readonly context: Session.Context, fallbackToCpuOps: boolean) {
    // this.opResolveRules = fallbackToCpuOps ? WASM_OP_RESOLVE_RULES.concat(CPU_OP_RESOLVE_RULES) :
    // WASM_OP_RESOLVE_RULES;
  }

  createInferenceHandler(): InferenceHandler {
    return new WasmInferenceHandler(this, this.context.profiler);
  }

  onGraphInitialized(graph: Graph, opset: ReadonlyArray<OpSet>) {
    // STEP.1 inference types
    const types = graph.getValueTypes().slice();
    const ops = graph.getNodes().map(node => resolveOperator<OperatorInfo>(node, opset, OP_INFO_RESOLVE_RULES));
    ops.forEach((op, i) => {
      const node = graph.getNodes()[i];
      const inputTypes = node.inputs.map(i => types[i]);

      let undefinedTypeIndex = inputTypes.indexOf(0);
      if (undefinedTypeIndex !== -1) {
        throw new Error(
            `type inference failed: unexpected undefined type at input[${undefinedTypeIndex}] of node ${node.name}`);
      }
      const inferencedOutputTypes = op.inferenceType(inputTypes);
      undefinedTypeIndex = inferencedOutputTypes.indexOf(0);
      if (undefinedTypeIndex !== -1) {
        throw new Error(
            `type inference failed: unexpected undefined type at output[${undefinedTypeIndex}] of node ${node.name}`);
      }
      if (inferencedOutputTypes.length !== node.outputs.length) {
        throw new Error(`type inference failed: length mismatch for output of node ${node.name}`);
      }
      inferencedOutputTypes.forEach((t, i) => {
        if (types[node.outputs[i]] !== 0 && types[node.outputs[i]] !== t) {
          throw new Error(
              `type inference failed: conflict inference result at output[${undefinedTypeIndex}] of node ${node.name}`);
        }
        types[node.outputs[i]] = t;
      });
    });

    // STEP.2 create inference context
    this.bindingInstance = getInstance();
    this.wasmContext =
        new (this.bindingInstance.InferenceContext)(graph.getNodes().length, graph.getValues().length, types);

    // STEP.3 set initializers
    graph.getValues().forEach((v, i) => {
      if (v.from === -1 && v.tensor) {
        this.wasmContext.setInitializer(i, v.tensor.dims);
        this.uploadData(v.tensor.numberData, this.wasmContext.getTensorData(i), this.wasmContext.getTensorDataSize(i));
      }
    });

    // STEP.4 set attributes
    ops.forEach((op, i) => {
      const attributes = graph.getNodes()[i].attributes;
      op.initializeAttributes(attributes);
      attributes.forEach((name, type, value) => {
        switch (type) {
          case 'float':
            this.wasmContext.addAttribute_f(i, name, value as number);
            break;
          case 'floats':
            this.wasmContext.addAttribute_floats(i, name, value as number[]);
            break;
          case 'int':
            this.wasmContext.addAttribute_i(i, name, safeInt32(value as number));
            break;
          case 'ints':
            this.wasmContext.addAttribute_ints(i, name, (value as number[]).map(safeInt32));
            break;
          case 'string':
            this.wasmContext.addAttribute_s(i, name, (value as string));
            break;
          case 'strings':
            this.wasmContext.addAttribute_strings(i, name, (value as string[]));
            break;
          default:
            throw new Error(`unsupported attribute type: ${type}`);
        }
      });
    });

    // STEP.5 init kernels
    ops.forEach((op, i) => {
      const node = graph.getNodes()[i];
      this.wasmContext.initKernel(
          i, node.opType, opset[0].domain, opset[0].version, node.inputs, node.outputs, op.hash);
    });

    // ExecutionPlan implementation
    this.customExecutionPlan = {
      execute(sessionHandler: WasmSessionHandler, modelInputs: Tensor[]): Promise<Tensor[]> {
        for (let i = 0; i < modelInputs.length; i++) {
          const valueIndex = graph.getInputIndices()[i];
          sessionHandler.wasmContext.setInput(valueIndex, modelInputs[i].dims);
          sessionHandler.uploadData(
              modelInputs[i].numberData, sessionHandler.wasmContext.getTensorData(valueIndex),
              sessionHandler.wasmContext.getTensorDataSize(valueIndex));
        }

        sessionHandler.wasmContext.run();

        const outputs = [];
        for (let i = 0; i < graph.getOutputIndices().length; i++) {
          const valueIndex = graph.getOutputIndices()[i];

          const tensor = new Tensor(
              sessionHandler.wasmContext.getTensorShape(valueIndex),
              ProtoUtil.tensorDataTypeFromProto(types[valueIndex]));
          sessionHandler.downloadData(
              tensor.numberData, sessionHandler.wasmContext.getTensorData(valueIndex),
              sessionHandler.wasmContext.getTensorDataSize(valueIndex));
          outputs.push(tensor);
        }
        return Promise.resolve(outputs);
      },

    };
  }

  uploadData(data: Tensor.NumberType, byteOffset: number, length: number): void {
    new (data.constructor as Float32ArrayConstructor)(this.bindingInstance.HEAPU8.buffer, byteOffset, length).set(data);
  }

  downloadData(data: Tensor.NumberType, byteOffset: number, length: number): void {
    data.set(new (data.constructor as Float32ArrayConstructor)(this.bindingInstance.HEAPU8.buffer, byteOffset, length));
  }

  resolve(node: Graph.Node, opsets: ReadonlyArray<OpSet>, graph: Graph): Operator {
    // const op = resolveOperator(node, opsets, this.opResolveRules);
    // op.initialize(node.attributes, node, graph);
    // return op;
    throw new Error('should not run into here');
  }

  // vNEXT latest:
  ortInit: boolean;
  sessionHandle: number;

  inputNames: string[];
  inputNamesUTF8Encoded: number[];
  outputNames: string[];
  outputNamesUTF8Encoded: number[];

  loadModel(model: Uint8Array) {
    const wasm = getInstance();
    if (!this.ortInit) {
      wasm._ort_init();
      this.ortInit = true;
    }

    const modelDataOffset = wasm._malloc(model.byteLength);
    try {
      wasm.HEAPU8.set(model, modelDataOffset);
      this.sessionHandle = wasm._ort_create_session(modelDataOffset, model.byteLength);
    } finally {
      wasm._free(modelDataOffset);
    }

    const inputCount = wasm._ort_get_input_count(this.sessionHandle);
    const outputCount = wasm._ort_get_output_count(this.sessionHandle);

    this.inputNames = [];
    this.inputNamesUTF8Encoded = [];
    this.outputNames = [];
    this.outputNamesUTF8Encoded = [];
    for (let i = 0; i < inputCount; i++) {
      const name = wasm._ort_get_input_name(this.sessionHandle, i);
      this.inputNamesUTF8Encoded.push(name);
      this.inputNames.push(wasm.UTF8ToString(name));
    }
    for (let i = 0; i < outputCount; i++) {
      const name = wasm._ort_get_output_name(this.sessionHandle, i);
      this.outputNamesUTF8Encoded.push(name);
      this.outputNames.push(wasm.UTF8ToString(name));
    }
  }

  run(inputs: Map<string, Tensor>|Tensor[]): Map<string, Tensor> {
    const wasm = getInstance();

    let inputIndices: number[] = [];
    if (!Array.isArray(inputs)) {
      const inputArray: Tensor[] = [];
      inputs.forEach((tensor, name) => {
        const index = this.inputNames.indexOf(name);
        if (index === -1) {
          throw new Error(`invalid input '${name}'`);
        }
        inputArray.push(tensor);
        inputIndices.push(index);
      });
      inputs = inputArray;
    } else {
      inputIndices = inputs.map((t, i) => i);
    }

    const inputCount = inputs.length;
    const outputCount = this.outputNames.length;

    const inputValues: number[] = [];
    const inputDataOffsets: number[] = [];
    // create input tensors
    for (let i = 0; i < inputCount; i++) {
      const data = inputs[i].numberData;
      const dataOffset = wasm._malloc(data.byteLength);
      inputDataOffsets.push(dataOffset);
      wasm.HEAPU8.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), dataOffset);

      const dims = inputs[i].dims;

      const stack = wasm.stackSave();
      const dimsOffset = wasm.stackAlloc(4 * dims.length);
      try {
        let dimIndex = dimsOffset / 4;
        dims.forEach(d => wasm.HEAP32[dimIndex++] = d);
        const tensor = wasm._ort_create_tensor(
            ProtoUtil.tensorDataTypeStringToEnum(inputs[i].type), dataOffset, data.byteLength, dimsOffset, dims.length);
        inputValues.push(tensor);
      } finally {
        wasm.stackRestore(stack);
      }
    }

    const beforeRunStack = wasm.stackSave();
    const inputValuesOffset = wasm.stackAlloc(inputCount * 4);
    const inputNamesOffset = wasm.stackAlloc(inputCount * 4);
    const outputValuesOffset = wasm.stackAlloc(outputCount * 4);
    const outputNamesOffset = wasm.stackAlloc(outputCount * 4);
    try {
      let inputValuesIndex = inputValuesOffset / 4;
      let inputNamesIndex = inputNamesOffset / 4;
      let outputValuesIndex = outputValuesOffset / 4;
      let outputNamesIndex = outputNamesOffset / 4;
      for (let i = 0; i < inputCount; i++) {
        wasm.HEAP32[inputValuesIndex++] = inputValues[i];
        wasm.HEAP32[inputNamesIndex++] = this.inputNamesUTF8Encoded[i];
      }
      for (let i = 0; i < outputCount; i++) {
        wasm.HEAP32[outputValuesIndex++] = 0;
        wasm.HEAP32[outputNamesIndex++] = this.outputNamesUTF8Encoded[i];
      }

      wasm._ort_run(
          this.sessionHandle, inputNamesOffset, inputValuesOffset, inputCount, outputNamesOffset, outputCount,
          outputValuesOffset);

      const output = new Map<string, Tensor>();

      for (let i = 0; i < outputCount; i++) {
        const tensor = wasm.HEAPU32[outputValuesOffset / 4 + i];

        const beforeGetTensorDataStack = wasm.stackSave();
        // stack allocate 4 pointer value
        const tensorDataOffset = wasm.stackAlloc(4 * 4);
        try {
          wasm._ort_get_tensor_data(
              tensor, tensorDataOffset, tensorDataOffset + 4, tensorDataOffset + 8, tensorDataOffset + 12);
          let tensorDataIndex = tensorDataOffset / 4;
          const dataType = wasm.HEAPU32[tensorDataIndex++];
          const dataOffset = wasm.HEAPU32[tensorDataIndex++];
          const dimsOffset = wasm.HEAPU32[tensorDataIndex++];
          const dimsLength = wasm.HEAPU32[tensorDataIndex++];
          const dims = [];
          for (let i = 0; i < dimsLength; i++) {
            dims.push(wasm.HEAPU32[dimsOffset / 4 + i]);
          }
          wasm._ort_free(dimsOffset);

          const t = new Tensor(dims, ProtoUtil.tensorDataTypeFromProto(dataType));
          new Uint8Array(t.numberData.buffer, t.numberData.byteOffset, t.numberData.byteLength)
              .set(wasm.HEAPU8.subarray(dataOffset, dataOffset + t.numberData.byteLength));
          output.set(this.outputNames[i], t);
        } finally {
          wasm.stackRestore(beforeGetTensorDataStack);
        }

        wasm._ort_release_tensor(tensor);
      }

      inputValues.forEach(t => wasm._ort_release_tensor(t));
      inputDataOffsets.forEach(i => wasm._free(i));

      return output;
    } finally {
      wasm.stackRestore(beforeRunStack);
    }
  }
  dispose() {
    const wasm = getInstance();
    if (this.inputNamesUTF8Encoded) {
      this.inputNamesUTF8Encoded.forEach(str => wasm._ort_free(str));
      this.inputNamesUTF8Encoded = [];
    }
    if (this.outputNamesUTF8Encoded) {
      this.outputNamesUTF8Encoded.forEach(str => wasm._ort_free(str));
      this.outputNamesUTF8Encoded = [];
    }
    if (this.sessionHandle) {
      wasm._ort_release_session(this.sessionHandle);
      this.sessionHandle = 0;
    }
  }
}

function safeInt32(num: number): number {
  return Math.min(2147483647, Math.max(num, -2147483648));
}
