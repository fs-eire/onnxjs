// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Tensor} from '../../../tensor';
// import {ShapeUtil} from '../../../util';
import {getGlsl} from '../glsl-source';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, RunData, WebGLOperator} from '../types';

export class WebGLUnpack implements WebGLOperator {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }
  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    const inputTexture = handler.getTextureData(inputs[0].dataId);
    if (!inputTexture) {
      throw new Error(`packed input texture must exist`);
    }
    const outputLayout = handler.createTextureLayoutFromShape(inputTexture.unpackedShape);
    const outputShape = outputLayout.shape;
    const rank = outputShape.length;

    const glsl = getGlsl(handler.session.backend.glContext.version);
    let shaderSource = `
    vec4 process(int rc[${rank}]) {
      return (vec4(getColorAsFloat(${glsl.texture2D}(A, vec2(0.5,0.5))), 0., 0., 0.));
    }
    `;
    if (outputShape.length > 0) {
      const setup = getSetup(rank, inputTexture.width, inputTexture.height);
      shaderSource = `
        float process(int m[${rank}]) {
            ${setup}
            vec4 result = ${glsl.texture2D}(A, coords);

            return isEvenC ? (isEvenR ? result.x : result.z) : (isEvenR ? result.y : result.w);
        }
        `;
    }
    return {
      inputLayouts: [handler.getOrCreateTextureLayout(inputs[0])],
      outputLayout,
      samplers: ['A'],
      shaderSource,
      isInputsPacked: true,
      isOutputPacked: false,
    };
  }
  createRunData(handler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    const inputTDs = [handler.getOrCreateTextureData(inputs[0], programInfo.inputLayouts[0])];
    return {
      inputTextureDatas: inputTDs,
      outputTextureData: handler.createTextureDataFromLayout(programInfo.outputLayout, inputTDs[0].tensor.type),
      uniformData: {}
    };
  }
}

function getSetup(rank: number, textureWidth: number, textureHeight: number): string {
  let setup = `int sm[${rank}];`;
  for (let i = 0; i < rank - 2; i++) {
    setup += `sm[${i}] = m[${i}];`
  }
  if (rank === 1) {
    setup += `
    bool isEvenC = (m[0] % 2 == 0);
    bool isEvenR = true;
    sm[0] = (m[0]) / 2;
    int offset = indicesToOffset_A(sm);
    vec2 coords = offsetToCoords(offset, ${textureWidth}, ${textureHeight});`;
  } else {
    setup += `
    bool isEvenC = (m[${rank - 1}] % 2 == 0);
    bool isEvenR = (m[${rank - 2}] % 2 == 0);
    sm[${rank - 1}] = (m[${rank - 1}]) / 2;
    sm[${rank - 2}] = (m[${rank - 2}]) / 2;
    int offset = indicesToOffset_A(sm);
    vec2 coords = offsetToCoords(offset, ${textureWidth}, ${textureHeight});`;
  }
  return setup;
}
