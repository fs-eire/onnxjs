// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {env} from '../../env';

import * as DataEncoders from './texture-data-encoder';
import {DataEncoder, Encoder} from './texture-data-encoder';
import {Disposable} from './utils';

/**
 * Abstraction and wrapper around WebGLRenderingContext and its operations
 */
export class WebGLContext implements Disposable {
  gl: WebGLRenderingContext;
  version: 1|2;

  private vertexbuffer: WebGLBuffer;
  private framebuffer: WebGLFramebuffer;

  // WebGL flags and vital parameters
  floatDownloadEnabled: boolean;
  renderFloat32Enabled: boolean;
  maxTextureSize: number;
  // private maxCombinedTextureImageUnits: number;
  private maxTextureImageUnits: number;
  // private maxCubeMapTextureSize: number;
  // private shadingLanguageVersion: string;
  // private webglVendor: string;
  // private webglVersion: string;

  // WebGL2 flags and vital parameters
  // private max3DTextureSize: number;
  // private maxArrayTextureLayers: number;
  // private maxColorAttachments: number;
  // private maxDrawBuffers: number;

  // WebGL extensions
  textureFloatExtension: OES_texture_float|null;
  textureHalfFloatExtension: OES_texture_half_float|null;

  // WebGL2 extensions
  colorBufferFloatExtension: {}|null;

  private disposed: boolean;
  private frameBufferBound = false;

  constructor(gl: WebGLRenderingContext, version: 1|2) {
    this.gl = gl;
    this.version = version;

    this.getExtensions();
    this.vertexbuffer = this.createVertexbuffer();
    this.framebuffer = this.createFramebuffer();
    this.queryVitalParameters();
  }

  allocateTexture(width: number, height: number, encoder: DataEncoder, data?: Encoder.DataArrayType): WebGLTexture {
    const gl = this.gl;
    // create the texture
    const texture = gl.createTexture();
    // bind the texture so the following methods effect this texture.
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const buffer = data ? encoder.encode(data, width * height) : null;
    gl.texImage2D(
        gl.TEXTURE_2D,
        0,  // Level of detail.
        encoder.internalFormat, width, height,
        0,  // Always 0 in OpenGL ES.
        encoder.format, encoder.textureType, buffer);
    this.checkError();
    return texture as WebGLTexture;
  }
  updateTexture(
      texture: WebGLTexture, width: number, height: number, dataType: Encoder.DataType, channels: number,
      data: Encoder.DataArrayType): void {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    const encoder = this.getEncoder(dataType, channels);
    const buffer = encoder.encode(data, width * height);
    gl.texSubImage2D(
        gl.TEXTURE_2D,
        0,  // level
        0,  // xoffset
        0,  // yoffset
        width, height, encoder.format, encoder.textureType, buffer);
    this.checkError();
  }
  attachFramebuffer(texture: WebGLTexture, width: number, height: number): void {
    const gl = this.gl;
    // Make it the target for framebuffer operations - including rendering.
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
    gl.framebufferTexture2D(
        gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture,
        0);  // 0, we aren't using MIPMAPs
    this.checkError();
    gl.viewport(0, 0, width, height);
    gl.scissor(0, 0, width, height);
  }
  readTexture(
      texture: WebGLTexture, width: number, height: number, dataSize: number, dataType: Encoder.DataType,
      channels: number): Encoder.DataArrayType {
    const gl = this.gl;
    if (!channels) {
      channels = 1;
    }
    if (!this.frameBufferBound) {
      this.attachFramebuffer(texture, width, height);
    }
    const encoder = this.getEncoder(dataType, channels);
    const buffer = encoder.allocate(width * height);
    // bind texture to framebuffer
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.framebufferTexture2D(
        gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture,
        0);  // 0, we aren't using MIPMAPs
    // TODO: Check if framebuffer is ready
    gl.readPixels(0, 0, width, height, gl.RGBA, encoder.textureType, buffer);
    this.checkError();
    // unbind FB
    return encoder.decode(buffer, dataSize);
  }
  isFramebufferReady(): boolean {
    // TODO: Implement logic to check if the framebuffer is ready
    return true;
  }
  getActiveTexture(): string {
    const gl = this.gl;
    const n = gl.getParameter(this.gl.ACTIVE_TEXTURE);
    return `TEXTURE${(n - gl.TEXTURE0)}`;
  }
  getTextureBinding(): WebGLTexture {
    return this.gl.getParameter(this.gl.TEXTURE_BINDING_2D);
  }
  getFramebufferBinding(): WebGLFramebuffer {
    return this.gl.getParameter(this.gl.FRAMEBUFFER_BINDING);
  }
  setVertexAttributes(positionHandle: number, textureCoordHandle: number): void {
    const gl = this.gl;
    gl.vertexAttribPointer(positionHandle, 3, gl.FLOAT, false, 20, 0);
    gl.enableVertexAttribArray(positionHandle);
    if (textureCoordHandle !== -1) {
      gl.vertexAttribPointer(textureCoordHandle, 2, gl.FLOAT, false, 20, 12);
      gl.enableVertexAttribArray(textureCoordHandle);
    }
    this.checkError();
  }
  createProgram(
      vertexShader: WebGLShader,
      fragShader: WebGLShader,
      ): WebGLProgram {
    const gl = this.gl;
    const program = gl.createProgram()!;

    // the program consists of our shaders
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragShader);
    gl.linkProgram(program);
    return program;
  }
  compileShader(shaderSource: string, shaderType: number): WebGLShader {
    const gl = this.gl;
    const shader = gl.createShader(shaderType);
    if (!shader) {
      throw new Error(`createShader() returned null with type ${shaderType}`);
    }

    gl.shaderSource(shader, shaderSource);
    gl.compileShader(shader);
    if (gl.getShaderParameter(shader, gl.COMPILE_STATUS) === false) {
      throw new Error(`Failed to compile shader: ${gl.getShaderInfoLog(shader)}`);
    }
    return shader;
  }
  deleteShader(shader: WebGLShader): void {
    this.gl.deleteShader(shader);
  }
  bindTextureToUniform(texture: WebGLTexture, position: number, uniformHandle: WebGLUniformLocation): void {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + position);
    this.checkError();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    this.checkError();
    gl.uniform1i(uniformHandle, position);
    this.checkError();
  }
  draw(): void {
    this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
    this.checkError();
  }
  checkError(): void {
    if (env.debug) {
      const gl = this.gl;
      const error = gl.getError();
      let label = '';
      switch (error) {
        case (gl.NO_ERROR):
          return;
        case (gl.INVALID_ENUM):
          label = 'INVALID_ENUM';
          break;
        case (gl.INVALID_VALUE):
          label = 'INVALID_VALUE';
          break;
        case (gl.INVALID_OPERATION):
          label = 'INVALID_OPERATION';
          break;
        case (gl.INVALID_FRAMEBUFFER_OPERATION):
          label = 'INVALID_FRAMEBUFFER_OPERATION';
          break;
        case (gl.OUT_OF_MEMORY):
          label = 'OUT_OF_MEMORY';
          break;
        case (gl.CONTEXT_LOST_WEBGL):
          label = 'CONTEXT_LOST_WEBGL';
          break;
        default:
          label = 'Unknown WebGL Error: ' + error.toString(16);
      }
      throw new Error(label);
    }
  }
  deleteTexture(texture: WebGLTexture): void {
    this.gl.deleteTexture(texture);
  }
  deleteProgram(program: WebGLProgram): void {
    this.gl.deleteProgram(program);
  }
  getEncoder(dataType: Encoder.DataType, channels: number, usage: Encoder.Usage = Encoder.Usage.Default): DataEncoder {
    if (this.version === 2) {
      return new DataEncoders.RedFloat32DataEncoder(this.gl as WebGL2RenderingContext, channels);
    }

    switch (dataType) {
      case 'float':
        if (usage === Encoder.Usage.UploadOnly || this.renderFloat32Enabled) {
          return new DataEncoders.RGBAFloatDataEncoder(this.gl, channels);
        } else {
          return new DataEncoders.RGBAFloatDataEncoder(
              this.gl, channels, this.textureHalfFloatExtension!.HALF_FLOAT_OES);
        }
      case 'int':
        throw new Error('not implemented');
      case 'byte':
        return new DataEncoders.Uint8DataEncoder(this.gl, channels);
      default:
        throw new Error(`Invalid dataType: ${dataType}`);
    }
  }
  clearActiveTextures(): void {
    const gl = this.gl;
    for (let unit = 0; unit < this.maxTextureImageUnits; ++unit) {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, null);
    }
  }
  dispose(): void {
    if (this.disposed) {
      return;
    }
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteFramebuffer(this.framebuffer);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.deleteBuffer(this.vertexbuffer);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
    gl.finish();
    this.disposed = true;
  }

  private createDefaultGeometry(): Float32Array {
    // Sets of x,y,z(=0),s,t coordinates.
    return new Float32Array([
      -1.0, 1.0,  0.0, 0.0, 1.0,  // upper left
      -1.0, -1.0, 0.0, 0.0, 0.0,  // lower left
      1.0,  1.0,  0.0, 1.0, 1.0,  // upper right
      1.0,  -1.0, 0.0, 1.0, 0.0
    ]);  // lower right
  }
  private createVertexbuffer(): WebGLBuffer {
    const gl = this.gl;
    const buffer = gl.createBuffer();
    if (!buffer) {
      throw new Error('createBuffer() returned null');
    }
    const geometry = this.createDefaultGeometry();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, geometry, gl.STATIC_DRAW);
    this.checkError();
    return buffer;
  }
  private createFramebuffer(): WebGLFramebuffer {
    const fb = this.gl.createFramebuffer();
    if (!fb) {
      throw new Error('createFramebuffer returned null');
    }
    return fb;
  }

  private queryVitalParameters(): void {
    const gl = this.gl;

    this.floatDownloadEnabled = this.isFloatDownloadEnabled();
    this.renderFloat32Enabled = this.isRenderFloat32Enabled();

    if (this.version === 1 && !this.textureHalfFloatExtension && !this.renderFloat32Enabled) {
      throw new Error(`both float32 and float16 TextureType are not supported`);
    }

    // this.maxCombinedTextureImageUnits = gl.getParameter(gl.MAX_COMBINED_TEXTURE_IMAGE_UNITS);
    this.maxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    this.maxTextureImageUnits = gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS);
    // this.maxCubeMapTextureSize = gl.getParameter(gl.MAX_CUBE_MAP_TEXTURE_SIZE);
    // this.shadingLanguageVersion = gl.getParameter(gl.SHADING_LANGUAGE_VERSION);
    // this.webglVendor = gl.getParameter(gl.VENDOR);
    // this.webglVersion = gl.getParameter(gl.VERSION);

    if (this.version === 2) {
      // this.max3DTextureSize = gl.getParameter(WebGL2RenderingContext.MAX_3D_TEXTURE_SIZE);
      // this.maxArrayTextureLayers = gl.getParameter(WebGL2RenderingContext.MAX_ARRAY_TEXTURE_LAYERS);
      // this.maxColorAttachments = gl.getParameter(WebGL2RenderingContext.MAX_COLOR_ATTACHMENTS);
      // this.maxDrawBuffers = gl.getParameter(WebGL2RenderingContext.MAX_DRAW_BUFFERS);
    }
  }
  private getExtensions(): void {
    if (this.version === 2) {
      this.colorBufferFloatExtension = this.gl.getExtension('EXT_color_buffer_float');
    } else {
      this.textureFloatExtension = this.gl.getExtension('OES_texture_float');
      this.textureHalfFloatExtension = this.gl.getExtension('OES_texture_half_float');
    }
  }

  private isRenderFloat32Enabled(): boolean {
    if (this.version === 2) {
      if (!this.colorBufferFloatExtension) {
        return false;
      }
    } else {
      if (!this.textureFloatExtension) {
        return false;
      }
    }
    return this.isFloatTextureAttachableToFrameBuffer();
  }
  private isFloatDownloadEnabled(): boolean {
    if (this.version === 2) {
      if (!this.colorBufferFloatExtension) {
        return false;
      }
    } else {
      if (!this.textureFloatExtension) {
        return false;
      }
      if (!this.gl.getExtension('WEBGL_color_buffer_float')) {
        return false;
      }
    }
    return this.isFloatTextureAttachableToFrameBuffer();
  }
  private isFloatTextureAttachableToFrameBuffer(): boolean {
    // test whether it is supported:
    // STEP.1 create a float texture
    const gl = this.gl;
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    const internalFormat = this.version === 2 ? (gl as unknown as {RGBA32F: number}).RGBA32F : gl.RGBA;
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, 1, 1, 0, gl.RGBA, gl.FLOAT, null);
    // STEP.2 bind a frame buffer
    const frameBuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, frameBuffer);
    // STEP.3 attach texture to framebuffer
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    // STEP.4 test whether framebuffer is complete
    const isComplete = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteTexture(texture);
    gl.deleteFramebuffer(frameBuffer);
    return isComplete;
  }
}
