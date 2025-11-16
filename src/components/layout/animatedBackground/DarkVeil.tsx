import React from "react";
import useDarkVeilGL from "./useDarkVeilGL";
import type {DarkVeilProps} from "./types";

export const DarkVeil: React.FC<DarkVeilProps> = (props) => {
    const canvasRef = useDarkVeilGL(props);

    return (
        <canvas
            ref={canvasRef}
            className="w-full h-full block"
        />
    );
};

export default DarkVeil;
