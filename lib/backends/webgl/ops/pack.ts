// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Tensor} from '../../../tensor';
// import {ShapeUtil} from '../../../util';
import {getGlsl} from '../glsl-source';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, RunData, WebGLOperator} from '../types';

export class WebGLPack implements WebGLOperator {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }
  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    const inputShape = inputs[0].dims;

    const outputLayout = handler.createTextureLayoutFromShape(inputShape, 4, inputShape, {isPacked: true});
    const outputShape = outputLayout.shape;
    const rank = outputShape.length;

    const glsl = getGlsl(handler.session.backend.glContext.version);
    let shaderSource = `
    vec4 process(int rc[${rank}]) {
      return (vec4(getColorAsFloat(${glsl.texture2D}(A, vec2(0.5,0.5))), 0., 0., 0.));
    }
    `;
    if (inputShape.length > 0) {
      // const outOfBoundsCondition = getOutOfBoundsCondition(rank, outputShape);
      const setup = getSetup(rank, outputShape, inputShape[inputShape.length - 1], inputShape[inputShape.length - 2]);
      shaderSource = `
        vec4 process(int m[${rank}]) {
            ${setup}

            return (vec4(x00,x01,x10,x11));
        }
        `;
    }
    return {
      inputLayouts: [handler.getOrCreateTextureLayout(inputs[0])],
      outputLayout,
      samplers: ['A'],
      shaderSource,
      isInputsPacked: false,
      isOutputPacked: true,
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

function getSetup(rank: number, outputShape: ReadonlyArray<number>, cols: number, rows: number): string {
  let setup = `int sm[${rank}];`;
  for (let i = 0; i < rank - 2; i++) {
    setup += `sm[${i}] = m[${i}];`
  }
  if (rank === 1) {
    setup += `
    int c = m[0] * 2;
    sm[0] = c;
    //int offset = indicesToOffset_A(sm);
    float x00 = _A(sm);
    float x01 = 0.0;
    if (c + 1 < ${cols}) {
      sm[0] = c + 1;
      x01 = _A(sm);
    }
    float x10 = 0.0;
    float x11 = 0.0;`;
  } else {
    setup += `
    int r = m[${rank - 2}] * 2;
    int c = m[${rank - 1}] * 2;
    int rp1 = r + 1;
    int cp1 = c + 1;

    bool cEdge = cp1 >= ${cols};
    bool rEdge = rp1 >= ${rows};

    sm[${rank - 2}] = r;
    sm[${rank - 1}] = c;
    //int offset = indicesToOffset_A(sm);

    float x00 = _A(sm);
    float x01 = 0.0;
    if (!cEdge) {
      sm[${rank - 1}] = cp1;
      x01 = _A(sm);
    }
    float x10 = 0.0;
    if (!rEdge) {
      sm[${rank - 2}] = rp1;
      sm[${rank - 1}] = c;
      x10 = _A(sm);
    }
    float x11 = 0.0;
    if (!rEdge && !cEdge) {
      sm[${rank - 1}] = cp1;
      x11 = _A(sm);
    }
  `;
  }
  return setup;
}
