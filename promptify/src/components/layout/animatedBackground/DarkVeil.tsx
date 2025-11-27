import React from "react";
import useDarkVeilGL from "./useDarkVeilGL";
import type {DarkVeilProps} from "./types";

export const DarkVeil: React.FC<DarkVeilProps> = (props) => {
    const { className, ...restProps } = props;
    const canvasRef = useDarkVeilGL(restProps);

    return (
        <canvas
            ref={canvasRef}
            className={`w-full h-full block${className ? ` ${className}` : ''}`}
        />
    );
};

export default DarkVeil;
