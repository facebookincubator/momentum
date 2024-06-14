/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React from 'react';
import clsx from 'clsx';
import styles from './HomepageFeatures.module.css';
import useBaseUrl from '@docusaurus/useBaseUrl';

const FeatureList = [
  {
    title: 'Forward and Inverse Kinematics with Interpretable Parameterization',
    image: '/img/momentum_1.png',
    description: (
      <>
      </>
    ),
  },
  {
    title: 'RGBD Body Tracking Solver',
    image: '/img/momentum_3.png',
    description: (
      <>
      </>
    ),
  },
  {
    title: 'Monocular RGB Body Tracking Solver',
    image: '/img/momentum_4.png',
    description: (
      <>
      </>
    ),
  },
];

function Feature({Svg, image, title, description}) {
  const imageUrl = useBaseUrl(image);
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        {Svg ? (
          <Svg className={styles.featureSvg} alt={title} />
        ) : (
          <img src={imageUrl} className={styles.featurePng} alt={title} />
        )}
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
