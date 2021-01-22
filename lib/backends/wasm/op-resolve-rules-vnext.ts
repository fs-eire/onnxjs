// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {OpSet} from '../../opset';

import {OperatorInfo} from './op-vnext';
import {BinaryOp} from './ops-vnext/binary-op';
import {Concat} from './ops-vnext/concat';
import {ElementWise} from './ops-vnext/element-wise';
import {Gather} from './ops-vnext/gather';
import {Gemm} from './ops-vnext/gemm';
import {MatMul} from './ops-vnext/matmul';
import {Reshape} from './ops-vnext/reshape';
import {Slice} from './ops-vnext/slice';
import {Unsqueeze} from './ops-vnext/unsqueeze';

export const OP_INFO_RESOLVE_RULES: ReadonlyArray<OpSet.ResolveRule<OperatorInfo>> = [
  ['Add', '', '7+', (node, opset) => new BinaryOp('Add', opset)],
  ['Concat', '', '4+', (node, opset) => new Concat(opset)],
  ['Gather', '', '1+', (node, opset) => new Gather(opset)],
  ['Gemm', '', '7+', (node, opset) => new Gemm(opset)],
  ['Slice', '', '1+', (node, opset) => new Slice(opset)],
  ['MatMul', '', '1+', (node, opset) => new MatMul(opset)],
  ['Mul', '', '7+', (node, opset) => new BinaryOp('Mul', opset)],
  ['Reshape', '', '5+', (node, opset) => new Reshape(opset)],
  ['Sigmoid', '', '5+', (node, opset) => new ElementWise('Sigmoid', opset)],
  ['Tanh', '', '6+', (node, opset) => new ElementWise('Tanh', opset)],
  ['Unsqueeze', '', '1+', (node, opset) => new Unsqueeze(opset)],
];
