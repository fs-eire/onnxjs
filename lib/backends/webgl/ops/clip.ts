// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Clip} from '../../../ops/clip';
import {Tensor} from '../../../tensor';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, RunData, WebGLOperator} from '../types';

export class WebGLClip extends Clip implements WebGLOperator {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }
  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    const outputShape = inputs[0].dims.slice();
    const shaderSource = `
      const float min = float(${this.min});
      const float max = float(${this.max});
      uniform sampler2D A;
      void main() {
        float v = texture2D(A, TexCoords).r;
        gl_FragColor = vec4(clamp(v, min, max));
      }
      `;
    return {
      hasMain: true,
      inputLayouts: [handler.createTextureLayout(inputs[0])],
      outputLayout: handler.createTextureLayout(outputShape),
      shaderSource,
    };
  }
  createRunData(handler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    const inputTDs = [handler.createTextureData(inputs[0], programInfo.inputLayouts[0])];
    return {
      inputTextureDatas: inputTDs,
      outputTextureData: handler.createTextureData(inputTDs[0].tensor.type, programInfo.outputLayout),
      uniformData: {}
    };
  }
}
