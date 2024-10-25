"use strict";(self.webpackChunkstaticdocs_starter=self.webpackChunkstaticdocs_starter||[]).push([[149],{70527:(e,n,i)=>{i.r(n),i.d(n,{assets:()=>s,contentTitle:()=>o,default:()=>u,frontMatter:()=>l,metadata:()=>d,toc:()=>a});var r=i(74848),t=i(28453);const l={sidebar_position:1},o="Getting Started",d={id:"user_guide/getting_started",title:"Getting Started",description:"This page guides you through the process of building Momentum and running the examples.",source:"@site/docs/02_user_guide/01_getting_started.md",sourceDirName:"02_user_guide",slug:"/user_guide/getting_started",permalink:"/momentum/docs/user_guide/getting_started",draft:!1,unlisted:!1,editUrl:"https://github.com/facebookincubator/momentum/edit/main/momentum/website/docs/02_user_guide/01_getting_started.md",tags:[],version:"current",sidebarPosition:1,frontMatter:{sidebar_position:1},sidebar:"tutorialSidebar",next:{title:"Creating Your Applications",permalink:"/momentum/docs/user_guide/creating_your_applications"}},s={},a=[{value:"Installing Momentum and PyMomentum",id:"installing-momentum-and-pymomentum",level:2},{value:"Pixi",id:"pixi",level:3},{value:"Conda",id:"conda",level:3},{value:"Micromamba",id:"micromamba",level:3},{value:"Building Momentum from Source",id:"building-momentum-from-source",level:2},{value:"Prerequisite",id:"prerequisite",level:3},{value:"Build and Test",id:"build-and-test",level:3},{value:"Hello World Example",id:"hello-world-example",level:3},{value:"Running Other Examples",id:"running-other-examples",level:3},{value:"Clean Up",id:"clean-up",level:3},{value:"FBX support (Windows only)",id:"fbx-support-windows-only",level:3}];function c(e){const n={a:"a",code:"code",h1:"h1",h2:"h2",h3:"h3",header:"header",li:"li",ol:"ol",p:"p",pre:"pre",ul:"ul",...(0,t.R)(),...e.components};return(0,r.jsxs)(r.Fragment,{children:[(0,r.jsx)(n.header,{children:(0,r.jsx)(n.h1,{id:"getting-started",children:"Getting Started"})}),"\n",(0,r.jsx)(n.p,{children:"This page guides you through the process of building Momentum and running the examples."}),"\n",(0,r.jsx)(n.h2,{id:"installing-momentum-and-pymomentum",children:"Installing Momentum and PyMomentum"}),"\n",(0,r.jsxs)(n.p,{children:["Momentum binary builds are available for Windows, macOS, and Linux via ",(0,r.jsx)(n.a,{href:"https://prefix.dev/",children:"Pixi"})," or the Conda package manager."]}),"\n",(0,r.jsx)(n.h3,{id:"pixi",children:"Pixi"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{children:"# Momentum (C++)\npixi add momentum-cpp\n\n# PyMomentum (Python)\npixi add pymomentum\n\n# Both\npixi add momentum\n"})}),"\n",(0,r.jsx)(n.h3,{id:"conda",children:"Conda"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{children:"conda install -c conda-forge momentum-cpp\nconda install -c conda-forge pymomentum # Linux only\nconda install -c conda-forge momentum\n"})}),"\n",(0,r.jsx)(n.h3,{id:"micromamba",children:"Micromamba"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{children:"micromamba install -c conda-forge momentum-cpp\nmicromamba install -c conda-forge pymomentum # Linux only\nmicromamba install -c conda-forge momentum\n"})}),"\n",(0,r.jsx)(n.h2,{id:"building-momentum-from-source",children:"Building Momentum from Source"}),"\n",(0,r.jsx)(n.h3,{id:"prerequisite",children:"Prerequisite"}),"\n",(0,r.jsx)(n.p,{children:"Complete the following steps only once:"}),"\n",(0,r.jsxs)(n.ol,{children:["\n",(0,r.jsxs)(n.li,{children:["\n",(0,r.jsxs)(n.p,{children:["Install Pixi by following the instructions on ",(0,r.jsx)(n.a,{href:"https://prefix.dev/",children:"https://prefix.dev/"})]}),"\n"]}),"\n",(0,r.jsxs)(n.li,{children:["\n",(0,r.jsx)(n.p,{children:"Clone the repository and navigate to the root directory:"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{children:"git clone https://github.com/facebookincubator/momentum\ncd momentum\n"})}),"\n",(0,r.jsx)(n.p,{children:"Ensure that all subsequent commands are executed in the project's root directory unless specified otherwise."}),"\n"]}),"\n"]}),"\n",(0,r.jsx)(n.h3,{id:"build-and-test",children:"Build and Test"}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:["\n",(0,r.jsx)(n.p,{children:"Build the project with the following command (note that the first run may take a few minutes as it installs all dependencies):"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{children:"pixi run build\n"})}),"\n"]}),"\n",(0,r.jsxs)(n.li,{children:["\n",(0,r.jsx)(n.p,{children:"Run the tests with:"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{children:"pixi run test\n"})}),"\n"]}),"\n"]}),"\n",(0,r.jsxs)(n.p,{children:["To view all available command lines, run ",(0,r.jsx)(n.code,{children:"pixi task list"}),"."]}),"\n",(0,r.jsx)(n.h3,{id:"hello-world-example",children:"Hello World Example"}),"\n",(0,r.jsxs)(n.p,{children:["To run the ",(0,r.jsx)(n.code,{children:"hello_world"})," example:"]}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{children:"pixi run hello_world\n"})}),"\n",(0,r.jsx)(n.p,{children:"Alternatively, you can directly run the executable:"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{children:"# Linux and macOS\n./build/hello_world\n\n# Windows\n./build/Release/hello_world.exe\n"})}),"\n",(0,r.jsx)(n.h3,{id:"running-other-examples",children:"Running Other Examples"}),"\n",(0,r.jsx)(n.p,{children:"To run other examples:"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{children:"pixi run glb_viewer --help\n"})}),"\n",(0,r.jsxs)(n.p,{children:["For more examples, please refer to the ",(0,r.jsx)(n.a,{href:"https://facebookincubator.github.io/momentum/docs/examples/viewers",children:"Examples"})," page."]}),"\n",(0,r.jsx)(n.h3,{id:"clean-up",children:"Clean Up"}),"\n",(0,r.jsx)(n.p,{children:"If you need to start over for any reason:"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{children:"pixi run clean\n"})}),"\n",(0,r.jsxs)(n.p,{children:["Momentum uses the ",(0,r.jsx)(n.code,{children:"build/"})," directory for CMake builds, and ",(0,r.jsx)(n.code,{children:".pixi/"})," for the Pixi virtual environment. You can clean up everything by either manually removing these directories or by running the command above."]}),"\n",(0,r.jsx)(n.h3,{id:"fbx-support-windows-only",children:"FBX support (Windows only)"}),"\n",(0,r.jsxs)(n.p,{children:["To load and save Autodesk's FBX file format, you need to install the FBX SDK 2019.2 from Autodesk's ",(0,r.jsx)(n.a,{href:"https://aps.autodesk.com/developer/overview/fbx-sdk",children:"website"})," or ",(0,r.jsx)(n.a,{href:"https://www.autodesk.com/content/dam/autodesk/www/adn/fbx/20192/fbx20192_fbxsdk_vs2017_win.exe",children:"this direct link"})," first. After installing the SDK, you can build with ",(0,r.jsx)(n.code,{children:"MOMENTUM_BUILD_IO_FBX=ON"}),":"]}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{children:'# Powershell\n$env:MOMENTUM_BUILD_IO_FBX = "ON"; pixi run <target>\n\n# cmd\nset MOMENTUM_BUILD_IO_FBX=ON && pixi run <target>\n'})}),"\n",(0,r.jsx)(n.p,{children:"For example, file conversion can be run as follows:"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{children:'# Powershell\n$env:MOMENTUM_BUILD_IO_FBX = "ON"; pixi run convert_model -d <input.glb> -o <out.fbx>\n\n# cmd\nset MOMENTUM_BUILD_IO_FBX=ON && pixi run convert_model -d <input.glb> -o <out.fbx>\n'})})]})}function u(e={}){const{wrapper:n}={...(0,t.R)(),...e.components};return n?(0,r.jsx)(n,{...e,children:(0,r.jsx)(c,{...e})}):c(e)}}}]);