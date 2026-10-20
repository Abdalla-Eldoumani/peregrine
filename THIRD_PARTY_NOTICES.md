# Third-party notices

Peregrine's compiled module and its wheels contain code from the projects
below. Each is used under the license reproduced or named here.

## nanobind

The Python bindings are built with nanobind, whose support code is compiled
into the extension module.

```
Copyright (c) 2022 Wenzel Jakob <wenzel.jakob@epfl.ch>, All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## tsl::robin_map

nanobind compiles in tsl::robin_map.

```
MIT License

Copyright (c) 2017 Thibaut Goetghebuer-Planchon <tessil@gmx.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## GNU OpenMP runtime (libgomp), Linux wheel

The Linux wheel bundles libgomp, the GCC OpenMP runtime, under the GNU General
Public License version 3 with the GCC Runtime Library Exception, version 3.1.
The exception permits distributing it with programs compiled by GCC under
terms of the distributor's choice. The license and the exception are published
at https://www.gnu.org/licenses/gpl-3.0.html and
https://www.gnu.org/licenses/gcc-exception-3.1.html, and the source is part of
GCC (https://gcc.gnu.org).

## Microsoft Visual C++ runtime (msvcp140.dll, vcomp140.dll), Windows wheel

The Windows wheel bundles the C++ standard library runtime (msvcp140.dll) and
the OpenMP runtime (vcomp140.dll) from the Microsoft Visual C++
Redistributable. Both are listed as distributable code in the Visual Studio
license terms and redistribution list
(https://learn.microsoft.com/en-us/visualstudio/releases/2022/redistribution),
and Microsoft's OpenMP library reference states that the OpenMP DLLs need to be
distributed with applications that use OpenMP. They remain subject to
Microsoft's license terms.
