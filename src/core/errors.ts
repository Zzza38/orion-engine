/** Base class for every error thrown by Orion Engine. */
export class OrionError extends Error {
    constructor(message: string, options?: { cause?: unknown }) {
        super(message, options);
        this.name = new.target.name;
    }
}

/** Thrown when tensor/matrix shapes do not line up. */
export class ShapeError extends OrionError {}

/** Thrown when a config, artifact, or argument is invalid. */
export class ValidationError extends OrionError {}

/** Thrown when a serialized model cannot be decoded. */
export class SerializationError extends OrionError {}

/** Thrown when training cannot continue, e.g. the loss became NaN or Infinity. */
export class TrainingError extends OrionError {}
