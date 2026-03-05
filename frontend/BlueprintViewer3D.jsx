import React, { useState, useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

export default function BlueprintViewer3D({ modelPath }) {
  const containerRef = useRef(null);
  const [scene] = useState(() => new THREE.Scene());
  const [camera, setCamera] = useState(null);
  const [renderer, setRenderer] = useState(null);

  useEffect(() => {
    if (!containerRef.current) return;

    scene.background = new THREE.Color(0xcccccc);

    const cameraInst = new THREE.PerspectiveCamera(
      75,
      containerRef.current.clientWidth / containerRef.current.clientHeight,
      0.1,
      1000
    );
    cameraInst.position.set(0, 0, 50);
    setCamera(cameraInst);

    const rendererInst = new THREE.WebGLRenderer({ antialias: true });
    rendererInst.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    rendererInst.shadowMap.enabled = true;
    containerRef.current.appendChild(rendererInst.domElement);
    setRenderer(rendererInst);

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 20, 10);
    directionalLight.castShadow = true;
    scene.add(directionalLight);

    const controls = new OrbitControls(cameraInst, rendererInst.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    if (modelPath) {
      const loader = new GLTFLoader();
      loader.load(modelPath, (gltf) => {
        scene.add(gltf.scene);
      });
    }

    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      rendererInst.render(scene, cameraInst);
    };
    animate();

    return () => {
      rendererInst.dispose();
      if (containerRef.current && rendererInst.domElement.parentNode === containerRef.current) {
        containerRef.current.removeChild(rendererInst.domElement);
      }
    };
  }, [modelPath, scene]);

  return (
    <div
      ref={containerRef}
      style={{ width: '100%', height: '100vh' }}
    />
  );
}
