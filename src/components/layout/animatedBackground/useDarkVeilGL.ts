import {useRef, useEffect} from 'react';
import {Renderer, Program, Mesh, Triangle, Vec2} from 'ogl';
import {DarkVeilProps} from './types';
import vertexShader from './shaders/darkVeil.vert.glsl';
import fragmentShader from './shaders/darkVeil.frag.glsl';

export default function useDarkVeilGL({
    hueShift = 0,
    speed = 0.5,
    warpAmount = 0,
    resolutionScale = 1,
}: DarkVeilProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) {
            return;
        }
        const parent = canvas.parentElement;
        if (!parent) {
            return;
        }

        const renderer = new Renderer({dpr: Math.min(window.devicePixelRatio, 2), canvas});
        const gl = renderer.gl;
        const geometry = new Triangle(gl);

        const program = new Program(gl, {
            vertex: vertexShader,
            fragment: fragmentShader,
            uniforms: {
                uTime: {value: 0},
                uResolution: {value: new Vec2()},
                uHueShift: {value: hueShift},
                uWarp: {value: warpAmount},
            },
        });

        const mesh = new Mesh(gl, {geometry, program});

        const resize = () => {
            const w = parent.clientWidth;
            const h = parent.clientHeight;
            renderer.setSize(w * resolutionScale, h * resolutionScale);
            program.uniforms.uResolution.value.set(w, h);
        };

        window.addEventListener('resize', resize);
        resize();

        const start = performance.now();
        let frame: number;

        const loop = () => {
            program.uniforms.uTime.value = ((performance.now() - start) / 1000) * speed;
            program.uniforms.uHueShift.value = hueShift;
            program.uniforms.uWarp.value = warpAmount;
            renderer.render({scene: mesh});
            frame = requestAnimationFrame(loop);
        };

        loop();

        return () => {
            cancelAnimationFrame(frame);
            window.removeEventListener('resize', resize);
        };
    }, [hueShift, speed, warpAmount, resolutionScale]);

    return canvasRef;
}
