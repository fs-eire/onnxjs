// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {InferenceHandler} from '../../backend';
import {Logger} from '../../instrument';
import {Tensor} from '../../tensor';
import {ShapeUtil} from '../../util';

import {WebGLUint8Encode} from './ops/uint8-encode';
import {ProgramManager} from './program-manager';
import {WebGLSessionHandler} from './session-handler';
import {Encoder} from './texture-data-encoder';
import {TextureHelper} from './texture-helper';
import {WidthHeightPrefs} from './texture-layout-strategy';
import {TextureData, TextureLayout, WebGLOperator} from './types';
import {getPackedShape} from './utils';

export class WebGLInferenceHandler implements InferenceHandler {
  textureHelper: TextureHelper;
  programManager: ProgramManager;
  private textureDataCache: Map<Tensor, TextureData>;
  constructor(public session: WebGLSessionHandler) {
    this.textureHelper = session.textureHelper;
    this.programManager = session.programManager;
    this.textureDataCache = new Map();
  }

  run(op: WebGLOperator, inputs: Tensor[]): Tensor[] {
    let artifact = this.programManager.getArtifact(op);
    if (!artifact) {
      const programInfo = op.createProgramInfo(this, inputs);
      artifact = this.programManager.build(programInfo);
      this.programManager.setArtifact(op, artifact);
    }
    const runData = op.createRunData(this, artifact.programInfo, inputs);
    this.programManager.run(artifact, runData);
    return [runData.outputTextureData.tensor];
  }

  /**
   * Create a TextureData object from a tensor.
   * Usage = Encoder.Usage.UploadOnly.
   * If a related texture data is found in cache, returns it;
   * Otherwise:
   *   Creates a new texture layout if not provided;
   *   Creates WebGLTexture with the layout;
   *   Upload tensor data to the texture;
   *   Creates a texture data object associated with the given tensor.
   */
  createTextureData(tensor: Tensor, layout?: TextureLayout): TextureData;
  /**
   * Create a TextureData object from the given data type and texture layout.
   * Usage = Encoder.Usage.Default.
   */
  createTextureData(dataType: Tensor.DataType, layout: TextureLayout): TextureData;
  /**
   * Create a TextureData object using the given data and bind to the given tensor.
   * Usage = Encoder.Usage.UploadOnly.
   * NOTE: this function is a hack for Conv implementation. should remove this function, after rewriting Conv
   * implementation by Graph.Transformer
   */
  createTextureData(dataType: Tensor.DataType, layout: TextureLayout, data: Tensor.NumberType, tensor: Tensor):
      TextureData;
  /**
   * Create a TextureData object, using the given texture.
   * This function does not create new texture. Usually used in scenarios using texture sharing. (eg. Reshape)
   */
  createTextureData(dataType: Tensor.DataType, layout: TextureLayout, texture: WebGLTexture): TextureData;
  createTextureData(
      arg0: Tensor|Tensor.DataType, layout?: TextureLayout, arg2?: Tensor.NumberType|WebGLTexture, tensor?: Tensor) {
    if (typeof arg0 !== 'string') {
      return this.getOrCreateTextureData(arg0, layout);
    }

    if (arg2 === undefined) {
      return this.createTextureDataFromLayout(arg0, layout!);
    } else if (ArrayBuffer.isView(arg2)) {
      return this.createTextureDataFromLayout(arg0, layout!, tensor, arg2, Encoder.Usage.UploadOnly);
    } else {
      return this.createTextureDataFromTexture(arg0, layout!, arg2);
    }
  }

  private getOrCreateTextureData(tensor: Tensor, layout?: TextureLayout): TextureData {
    let td = this.getTextureData(tensor);
    if (!td) {
      Logger.verbose('InferenceHandler', `Creating new TextureData for dims: [${tensor.dims}]`);
      if (!layout) {
        layout = this.createTextureLayoutFromShape(tensor.dims.slice());
      }
      // graph inputs or initializers
      td = this.createTextureDataFromLayout(tensor.type, layout, tensor, tensor.numberData, Encoder.Usage.UploadOnly);
    } else {
      Logger.verbose('InferenceHandler', `Retrieving TextureData from cache: [${tensor.dims}]`);
    }
    return td;
  }
  private createTextureDataFromLayout(
      dataType: Tensor.DataType, layout: TextureLayout, tensor?: Tensor, data?: Tensor.NumberType,
      usage?: Encoder.Usage): TextureData {
    Logger.verbose('InferenceHandler', `Creating TextureData: layout:[${JSON.stringify(layout)}]`);
    const texture = this.textureHelper.createTextureFromLayout(dataType, layout, data, usage);
    return this.createTextureDataFromTexture(dataType, layout, texture, tensor);
  }
  private createTextureDataFromTexture(
      dataType: Tensor.DataType, layout: TextureLayout, texture: WebGLTexture, tensor?: Tensor) {
    const textureData: TextureData = {
      ...layout,
      tensor: tensor ||
          new Tensor(
                  layout.unpackedShape, dataType,
                  (id: Tensor.Id) => {
                    return this.readTexture(textureData);
                  }),
      texture
    };
    this.setTextureData(textureData.tensor, textureData);
    return textureData;
  }

  getTextureData(tensor: Tensor): TextureData|undefined {
    return this.session.isInitializer(tensor) ? this.session.getTextureData(tensor) : this.textureDataCache.get(tensor);
  }
  setTextureData(tensor: Tensor, td: TextureData): void {
    if (this.session.isInitializer(tensor)) {
      this.session.setTextureData(tensor, td);
    } else {
      this.textureDataCache.set(tensor, td);
    }
  }

  /**
   * Create a TextureLayout object from a tensor. If a related texture data is found, returns the cached texture layout.
   */
  createTextureLayout(tensor: Tensor, channels?: 1|4, unpackedShape?: ReadonlyArray<number>): TextureLayout;
  /**
   * Create a TextureLayout object from shape.
   */
  createTextureLayout(
      shape: ReadonlyArray<number>, channels?: 1|4, unpackedShape?: ReadonlyArray<number>,
      prefs?: WidthHeightPrefs): TextureLayout;
  createTextureLayout(
      arg0: Tensor|ReadonlyArray<number>, channels: 1|4 = 1, unpackedShape?: ReadonlyArray<number>,
      prefs?: WidthHeightPrefs) {
    if (arg0 instanceof Tensor) {
      return this.getOrCreateTextureLayout(arg0, channels, unpackedShape);
    } else {
      return this.createTextureLayoutFromShape(arg0, channels, unpackedShape, prefs);
    }
  }

  private getOrCreateTextureLayout(tensor: Tensor, channels = 1, unpackedShape?: ReadonlyArray<number>): TextureLayout {
    const td = this.getTextureData(tensor);
    if (td) {
      return td;
    }
    return this.createTextureLayoutFromShape(
        channels === 1 ? tensor.dims.slice() : getPackedShape(tensor.dims.slice()), channels, unpackedShape);
  }
  private createTextureLayoutFromShape(
      shape: ReadonlyArray<number>, channels = 1, unpackedShape?: ReadonlyArray<number>,
      prefs?: WidthHeightPrefs): TextureLayout {
    const [width, height] = this.session.layoutStrategy.computeTextureWH(shape, prefs);
    let inferredDims = shape;
    if (shape.length === 0) {
      inferredDims = [1];
    }
    if (channels === 1) {
      // unpackedShape will take `shape` and not `inferredDims` so as to create a scalar Tensor if need be
      unpackedShape = shape;
    } else if (!unpackedShape) {
      throw new Error('Unpacked shape is needed when using channels > 1');
    }
    return {
      width,
      height,
      channels: channels ? channels : 1,
      shape: inferredDims,
      strides: ShapeUtil.computeStrides(inferredDims),
      unpackedShape
    };
  }

  dispose(): void {
    this.textureHelper.clearActiveTextures();
    this.textureDataCache.forEach(td => this.textureHelper.releaseTexture(td));
    this.textureDataCache = new Map();
  }

  readTexture(textureData: TextureData): Tensor.NumberType {
    if (this.session.backend.forceUint8Reads) {
      const op = new WebGLUint8Encode();
      const uint8TD = op.runInternal(this, textureData);
      return this.textureHelper.readUint8TextureAsFloat(uint8TD);
    }
    const values = this.textureHelper.readTexture(textureData, textureData.tensor.type, textureData.channels);
    return values;
  }
}
