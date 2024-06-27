/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');
const {fbContent, fbInternalOnly} = require('docusaurus-plugin-internaldocs-fb/internal');

// With JSDoc @type annotations, IDEs can provide config autocompletion
/** @type {import('@docusaurus/types').DocusaurusConfig} */
(module.exports = {
  title: 'Momentum',
  tagline: 'A library for human kinematic motion and numerical optimization solvers to apply human motion',
  url: 'https://facebookincubator.github.io',
  baseUrl: '/momentum/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'throw',
  favicon: 'img/favicon.ico',
  organizationName: 'facebookincubator',
  projectName: 'momentum',
  customFields: {
    fbRepoName: 'fbsource',
    ossRepoPath: 'arvr/libraries/momentum',
  },

  presets: [
    [
      'docusaurus-plugin-internaldocs-fb/docusaurus-preset',
      /** @type {import('docusaurus-plugin-internaldocs-fb').PresetOptions} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: fbContent({
            internal:
              'https://www.internalfb.com/code/fbsource/arvr/libraries/momentum/website',
            external:
              'https://github.com/facebookincubator/momentum/edit/main/momentum/website',
          }),
        },
        experimentalXRepoSnippets: {
          baseDir: '.',
        },
        staticDocsProject: 'Momentum',
        trackingFile: 'fbcode/staticdocs/WATCHED_FILES',
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
        gtag: {
          trackingID: 'G-NQKPMTK7XB',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      navbar: {
        title: 'Momentum',
        logo: {
          alt: 'Momentum Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'doc',
            docId: 'user_guide/getting_started',
            position: 'left',
            label: 'Docs',
          },
          {
            href: 'pathname:///doxygen/index.html',
            position: 'left',
            label: 'API',
          },
          {
            href: 'https://github.com/facebookincubator/momentum',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              {
                label: 'Getting Started',
                to: '/docs/user_guide/getting_started',
              },
              {
                label: 'Examples',
                to: '/docs/examples/viewers',
              },
              {
                label: 'Developer Guide',
                to: '/docs/developer_guide/development_environment',
              },
            ],
          },
          {
            title: 'Legal',
            // Please do not remove the privacy and terms, it's a legal requirement.
            items: [
              {
                label: 'Privacy',
                href: 'https://opensource.fb.com/legal/privacy/',
                target: '_blank',
                rel: 'noreferrer noopener',
              },
              {
                label: 'Terms',
                href: 'https://opensource.fb.com/legal/terms/',
                target: '_blank',
                rel: 'noreferrer noopener',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Meta Platforms, Inc. Built with Docusaurus.`,
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
      },
    }),
});
