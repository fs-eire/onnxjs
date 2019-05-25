// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Pad} from '../../../ops/pad';
import {Tensor} from '../../../tensor';
import {CpuInferenceHandler} from '../inference-handler';

export class CpuPad extends Pad {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    return [inputs[0]];
  }
}
