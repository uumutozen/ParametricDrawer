%%
%% This is file `report.cls',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% classes.dtx  (with options: `report')
%% 
%% This is a generated file.
%% 
%% The source is maintained by the LaTeX Project team and bug
%% reports for it can be opened at http://latex-project.org/bugs.html
%% (but please observe conditions on bug reports sent to that address!)
%% 
%% 
%% Copyright 1993-2015
%% The LaTeX3 Project and any individual authors listed elsewhere
%% in this file.
%% 
%% This file was generated from file(s) of the LaTeX base system.
%% --------------------------------------------------------------
%% 
%% It may be distributed and/or modified under the
%% conditions of the LaTeX Project Public License, either version 1.3c
%% of this license or (at your option) any later version.
%% The latest version of this license is in
%%    http://www.latex-project.org/lppl.txt
%% and version 1.3c or later is part of all distributions of LaTeX
%% version 2005/12/01 or later.
%% 
%% This file has the LPPL maintenance status "maintained".
%% 
%% This file may only be distributed together with a copy of the LaTeX
%% base system. You may however distribute the LaTeX base system without
%% such generated files.
%% 
%% The list of all files belonging to the LaTeX base distribution is
%% given in the file `manifest.txt'. See also `legal.txt' for additional
%% information.
%% 
%% The list of derived (unpacked) files belonging to the distribution
%% and covered by LPPL is defined by the unpacking scripts (with
%% extension .ins) which are part of the distribution.
%% \CharacterTable
%%  {Upper-case    \A\B\C\D\E\F\G\H\I\J\K\L\M\N\O\P\Q\R\S\T\U\V\W\X\Y\Z
%%   Lower-case    \a\b\c\d\e\f\g\h\i\j\k\l\m\n\o\p\q\r\s\t\u\v\w\x\y\z
%%   Digits        \0\1\2\3\4\5\6\7\8\9
%%   Exclamation   \!     Double quote  \"     Hash (number) \#
%%   Dollar        \$     Percent       \%     Ampersand     \&
%%   Acute accent  \'     Left paren    \(     Right paren   \)
%%   Asterisk      \*     Plus          \+     Comma         \,
%%   Minus         \-     Point         \.     Solidus       \/
%%   Colon         \:     Semicolon     \;     Less than     \<
%%   Equals        \=     Greater than  \>     Question mark \?
%%   Commercial at \@     Left bracket  \[     Backslash     \\
%%   Right bracket \]     Circumflex    \^     Underscore    \_
%%   Grave accent  \`     Left brace    \{     Vertical bar  \|
%%   Right brace   \}     Tilde         \~}
\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{deumathproject}[2024/04/24 v1.0 Standard LaTeX document class]
\usepackage{amssymb, amsmath, amsfonts}
\usepackage{graphicx}
\usepackage{setspace}
\usepackage[utf8]{inputenc}

\newcommand\@ptsize{}
\newif\if@restonecol
\newif\if@titlepage
\@titlepagetrue
\newif\if@openright

\DeclareOption{a4paper}
{
	\setlength\paperheight {297mm}
	\setlength\paperwidth  {210mm}
}
\DeclareOption{12pt}{\renewcommand\@ptsize{2}}
\DeclareOption{twoside}{\@twosidetrue  \@mparswitchtrue}
\DeclareOption{final}{\setlength\overfullrule{0pt}}
\DeclareOption{titlepage}{\@titlepagetrue}
\DeclareOption{openany}{\@openrightfalse}
\DeclareOption{onecolumn}{\@twocolumnfalse}

\ExecuteOptions{a4paper, 12pt, twoside, onecolumn, final, openany}
\ProcessOptions

\input{size12.clo}
\setlength\voffset{-1in} % https://en.wikibooks.org/wiki/LaTeX/Page_Layout
\setlength\hoffset{-1in}
\setlength\marginparwidth{0mm}
\setlength\oddsidemargin{30mm}
\setlength\evensidemargin{30mm}
\setlength\marginparsep{0mm}
\setlength\textwidth {165mm}
\setlength\topmargin{20mm}
\setlength\headheight{0mm}
\setlength\headsep{0mm}
\setlength\textheight {257mm}
\setlength\footskip{13mm} 
\setlength\parindent{5mm}
\setlength\parskip {\baselineskip}
\renewcommand{\baselinestretch}{1.5}

\renewcommand{\@afterindentfalse}{\@afterindenttrue}

\if@twoside
  \def\ps@headings{%
      \let\@oddfoot\@empty\let\@evenfoot\@empty
      \def\@evenhead{\thepage\hfil\slshape\leftmark}%
      \def\@oddhead{{\slshape\rightmark}\hfil\thepage}%
      \let\@mkboth\markboth
    \def\chaptermark##1{%
      \markboth {\MakeUppercase{%
        \ifnum \c@secnumdepth >\m@ne
            \@chapapp\ \thechapter. \ %
        \fi
        ##1}}{}}%
    \def\sectionmark##1{%
      \markright {\MakeUppercase{%
        \ifnum \c@ >\z@
          \thesection. \ %
        \fi
        ##1}}}}
\else
  \def\ps@headings{%
    \let\@oddfoot\@empty
    \def\@oddhead{{\slshape\rightmark}\hfil\thepage}%
    \let\@mkboth\markboth
    \def\chaptermark##1{%
      \markright {\MakeUppercase{%
        \ifnum \c@secnumdepth >\m@ne
            \@chapapp\ \thechapter. \ %
        \fi
        ##1}}}
    }
\fi
%\def\ps@myheadings{%
%    \let\@oddfoot\@empty\let\@evenfoot\@empty
%    \def\@evenhead{\thepage\hfil\slshape\leftmark}%
%    \def\@oddhead{{\slshape\rightmark}\hfil\thepage}%
%    \let\@mkboth\@gobbletwo
%    \let\chaptermark\@gobble
%    \let\sectionmark\@gobble
%    }

\def\reportkind#1{\gdef\@reportkind{#1}}    

%\if@titlepage
	\newcommand\maketitle{
		  \begin{titlepage}%
				\begin{center}
	%				\includegraphics[width=6cm]{DEU_Logo-English.pdf}\\
					$\;$\\
					\vspace{2cm}	
					\textbf{\fontsize{20}{\baselineskip}\selectfont Dokuz Eylül University}	\\
					\bigskip			
					\textbf{\fontsize{20}{\baselineskip}\selectfont Faculty of Science}	\\			
					\bigskip				
					\textbf{\fontsize{20}{\baselineskip}\selectfont Department of Mathematics}	\\			
					\vspace{3cm}
					\textbf{\fontsize{14}{\baselineskip}\selectfont {\@reportkind}}	\\
					\vspace{3cm}
					\textbf{\textit{\fontsize{24}{\baselineskip}\selectfont \MakeUppercase{\@title}}}\\
					\bigskip
					\textit{\textbf{\fontsize{14}{\baselineskip}\selectfont by}}\\				
					\bigskip				
					\textbf{\fontsize{14}{\baselineskip}\selectfont{\@author}}\\
					\vspace{5cm}				
					Project Supervisor\\				
					{\@supervisor}\\
					\vfill				
					{\@date}
				\end{center}
		  \end{titlepage}%
		  \setcounter{footnote}{0}%
		  \global\let\thanks\relax
		  \global\let\maketitle\relax
		  \global\let\@thanks\@empty
		  \global\let\@author\@empty
		  \global\let\@date\@empty
		  \global\let\@title\@empty
		  \global\let\title\relax
		  \global\let\author\relax
		  \global\let\date\relax
		  \global\let\and\relax
		  \pagenumbering{roman}
		  \setcounter{page}{2}
	}
%\else
%	\newcommand\maketitle{\par
%	  \begingroup
%	    \renewcommand\thefootnote{\@fnsymbol\c@footnote}%
%	    \def\@makefnmark{\rlap{\@textsuperscript{\normalfont\@thefnmark}}}%
%	    \long\def\@makefntext##1{\parindent 1em\noindent
%	            \hb@xt@1.8em{%
%	                \hss\@textsuperscript{\normalfont\@thefnmark}}##1}%
%	    \if@twocolumn
%	      \ifnum \col@number=\@ne
%	        \@maketitle
%	      \else
%	        \twocolumn[\@maketitle]%
%	      \fi
%	    \else
%	      \newpage
%	      \global\@topnum\z@   % Prevents figures from going at top of page.
%	      \@maketitle
%	    \fi
%	    \thispagestyle{plain}\@thanks
%	  \endgroup
%	  \setcounter{footnote}{0}%
%	  \global\let\thanks\relax
%	  \global\let\maketitle\relax
%	  \global\let\@maketitle\relax
%	  \global\let\@thanks\@empty
%	  \global\let\@author\@empty
%	  \global\let\@date\@empty
%	  \global\let\@title\@empty
%	  \global\let\title\relax
%	  \global\let\author\relax
%	  \global\let\date\relax
%	  \global\let\and\relax
%	}
%	\def\@maketitle{%
%	  \newpage
%	  \null
%	  \vskip 2em%
%	  \begin{center}%
%	  \let \footnote \thanks
%	    {\LARGE \@title \par}%
%	    \vskip 1.5em%
%	    {\large
%	      \lineskip .5em%
%	      \begin{tabular}[t]{c}%
%	        \@author
%	      \end{tabular}\par}%
%	    \vskip 1em%
%	    {\large \@date}%
%	  \end{center}%
%	  \par
%	  \vskip 1.5em}
%\fi


\def \dotdot{\protect\leaders\hbox{$\m@th
		\mkern \@dotsep mu\hbox{.}\mkern \@dotsep
		mu$}\hfill	
}


\newcommand*\chaptermark[1]{}
\setcounter{secnumdepth}{6}
\newcounter {chapter}
\newcounter {section}[chapter]
\newcounter {subsection}[section]
\newcounter {subsubsection}[subsection]
\newcounter {paragraph}[subsubsection]
\newcounter {subparagraph}[paragraph]
\renewcommand \thechapter {\@arabic\c@chapter}
\renewcommand \thesection {\thechapter.\@arabic\c@section}
\renewcommand\thesubsection   {\thesection.\@arabic\c@subsection}
\renewcommand\thesubsubsection{\thesubsection.\@arabic\c@subsubsection}
\renewcommand\theparagraph    {\thesubsubsection.\@arabic\c@paragraph}
\renewcommand\thesubparagraph {\theparagraph.\@arabic\c@subparagraph}
\newcommand\@chapapp{\chaptername}
\newcommand \theAsection {\thechapter.\@arabic\c@section}

\newcommand*\l@part[2]{%
	\ifnum \c@tocdepth >-2\relax
	\vspace{-41\p@}
	\addpenalty{-\@highpenalty}%
	\addvspace{2.25em \@plus\p@}%
	\setlength\@tempdima{3em}%
	\begingroup
	\parindent \z@ \rightskip \@pnumwidth
	\parfillskip -\@pnumwidth
	{\leavevmode
		#1\dotdot #2}\par
	\nobreak
	\global\@nobreaktrue
	\everypar{\global\@nobreakfalse\everypar{}}%
	\endgroup
	\fi}
	
\newcommand\chapter{\clearpage
                    \thispagestyle{plain}%
                    \global\@topnum\z@
                    \@afterindentfalse
                    \secdef\@chapter\@schapter}
                    
\def\@chapter[#1]#2{\ifnum \c@secnumdepth >\m@ne
                         \refstepcounter{chapter}%
                         \typeout{\@chapapp\space\thechapter.}%
                         \addcontentsline{toc}{chapter}%
                                   {\protect\numberline{\thechapter}#1}%
                    \else
                      \addcontentsline{toc}{chapter}{#1}%
                    \fi
                    \chaptermark{#1}%
                    \@makechapterhead{#2}%
                    \@afterheading
}

\def\@makechapterhead#1{%
  {\parindent \z@ \raggedright \normalfont
    \ifnum \c@secnumdepth >\m@ne
        \centering\bfseries \@chapapp\space \thechapter
        \par\nobreak
        \vspace{-15\p@}
    \fi
    \interlinepenalty\@M
    \bfseries \MakeUppercase{#1}\par\nobreak
    \vspace{8\p@}
  }}

\def\@schapter#1{\@makeschapterhead{#1}}

\def\@makeschapterhead#1{%
  \vspace*{50\p@}%
  {\parindent \z@ \raggedright
    \normalfont
    \interlinepenalty\@M
    \bfseries  #1\par\nobreak
    \vspace{-12\p@}
  }}

\newcommand\section{\@startsection {section}{1}{\z@}%
                                   {-3.5ex \@plus -1ex \@minus -.2ex}%
                                   {2.3ex \@plus.2ex}%
                                   {\normalfont\bfseries}}
\newcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {1.5ex \@plus .2ex}%
                                     {\normalfont\itshape\bfseries}}
\newcommand\subsubsection{\@startsection{subsubsection}{3}{.5cm}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {1.5ex \@plus .2ex}%
                                     {\normalfont\itshape}}
\newcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
                                    {3.25ex \@plus1ex \@minus.2ex}%
                                    {-1em}%
                                    {\normalfont\normalsize\bfseries}}
\newcommand\subparagraph{\@startsection{subparagraph}{5}{\parindent}%
                                       {3.25ex \@plus1ex \@minus .2ex}%
                                       {-1em}%
                                      {\normalfont\normalsize\bfseries}}
                                      
\if@twocolumn
  \setlength\leftmargini  {2em}
\else
  \setlength\leftmargini  {2.5em}
\fi

\leftmargin  \leftmargini
\setlength\leftmarginii  {2.2em}
\setlength\leftmarginiii {1.87em}
\setlength\leftmarginiv  {1.7em}

\if@twocolumn
  \setlength\leftmarginv  {.5em}
  \setlength\leftmarginvi {.5em}
\else
  \setlength\leftmarginv  {1em}
  \setlength\leftmarginvi {1em}
\fi

\setlength  \labelsep  {.5em}
\setlength  \labelwidth{\leftmargini}
\addtolength\labelwidth{-\labelsep}
\@beginparpenalty -\@lowpenalty
\@endparpenalty   -\@lowpenalty
\@itempenalty     -\@lowpenalty
\renewcommand\theenumi{\@arabic\c@enumi}
\renewcommand\theenumii{\@alph\c@enumii}
\renewcommand\theenumiii{\@roman\c@enumiii}
\renewcommand\theenumiv{\@Alph\c@enumiv}
\newcommand\labelenumi{\theenumi.}
\newcommand\labelenumii{(\theenumii)}
\newcommand\labelenumiii{\theenumiii.}
\newcommand\labelenumiv{\theenumiv.}
\renewcommand\p@enumii{\theenumi}
\renewcommand\p@enumiii{\theenumi(\theenumii)}
\renewcommand\p@enumiv{\p@enumiii\theenumiii}
\newcommand\labelitemi{\textbullet}
\newcommand\labelitemii{\normalfont\bfseries \textendash}
\newcommand\labelitemiii{\textasteriskcentered}
\newcommand\labelitemiv{\textperiodcentered}

\newenvironment{description}
               {\list{}{\labelwidth\z@ \itemindent-\leftmargin
                        \let\makelabel\descriptionlabel}}
               {\endlist}
               
\newcommand*\descriptionlabel[1]{\hspace\labelsep
                                \normalfont\bfseries #1}
%\if@titlepage
\newenvironment{abstract}{%
	\titlepage
	\null\vfil
	\@beginparpenalty\@lowpenalty
	\begin{center}%
		\bfseries \abstractname
		\@endparpenalty\@M
\end{center}}%
{\par\vfil\null\endtitlepage}
%\else
%\newenvironment{abstract}{%
%	\if@twocolumn
%		\section*{\abstractname}%
%	\else
%		\small
%		\begin{center}%
%			{\bfseries \abstractname\vspace{-.5em}\vspace{\z@}}%
%		\end{center}%
%		\quotation
%	\fi
%	}
%{\if@twocolumn\else\endquotation\fi}
%\fi

\newenvironment{verse}
               {\let\\\@centercr
                \list{}{\itemsep      \z@
                        \itemindent   -1.5em%
                        \listparindent\itemindent
                        \rightmargin  \leftmargin
                        \advance\leftmargin 1.5em}%
                \item\relax}
               {\endlist}
               
\newenvironment{quotation}
               {\list{}{\listparindent 1.5em%
                        \itemindent    \listparindent
                        \rightmargin   \leftmargin
                        \parsep        \z@ \@plus\p@}%
                \item\relax}
               {\endlist}
               
\newenvironment{quote}
               {\list{}{\rightmargin\leftmargin}%
                \item\relax}
               {\endlist}
               
\if@compatibility
	\newenvironment{titlepage}{%
	      \if@twocolumn
	        \@restonecoltrue\onecolumn
	      \else
	        \@restonecolfalse\newpage
	      \fi
	      \thispagestyle{empty}%
	      \setcounter{page}\z@
    }%
    {\if@restonecol\twocolumn \else \newpage \fi
    }
\else
	\newenvironment{titlepage}{%
	      \if@twocolumn
	        \@restonecoltrue\onecolumn
	      \else
	        \@restonecolfalse\newpage
	      \fi
	      \thispagestyle{empty}%
	      \setcounter{page}\@ne
    }%
    {\if@restonecol\twocolumn \else \newpage \fi
     \if@twoside\else
        \setcounter{page}\@ne
     \fi
    }
\fi

\newcommand\appendix{\par
	  \setcounter{chapter}{0}%
	  \setcounter{section}{0}%
	  \gdef\@chapapp{\appendixname}%
	  \gdef\thechapter{\@Alph\c@chapter}
}

\setlength\arraycolsep{5\p@}
\setlength\tabcolsep{6\p@}
\setlength\arrayrulewidth{.4\p@}
\setlength\doublerulesep{2\p@}
\setlength\tabbingsep{\labelsep}
\skip\@mpfootins = \skip\footins
\setlength\fboxsep{3\p@}
\setlength\fboxrule{.4\p@}
\@addtoreset {equation}{chapter}
\renewcommand\theequation
  {\ifnum \c@chapter>\z@ \thechapter.\fi \@arabic\c@equation}
\newcounter{figure}[chapter]
\renewcommand \thefigure
     {\ifnum \c@chapter>\z@ \thechapter.\fi \@arabic\c@figure}
\def\fps@figure{tbp}
\def\ftype@figure{1}
\def\ext@figure{lof}
\def\fnum@figure{\figurename\nobreakspace\thefigure}
\newenvironment{figure}
               {\@float{figure}}
               {\end@float}
\newenvironment{figure*}
               {\@dblfloat{figure}}
               {\end@dblfloat}
\newcounter{table}[chapter]
\renewcommand \thetable
     {\ifnum \c@chapter>\z@ \thechapter.\fi \@arabic\c@table}
\def\fps@table{tbp}
\def\ftype@table{2}
\def\ext@table{lot}
\def\fnum@table{\tablename\nobreakspace\thetable}
\newenvironment{table}
               {\@float{table}}
               {\end@float}
\newenvironment{table*}
               {\@dblfloat{table}}
               {\end@dblfloat}
\newlength\abovecaptionskip
\newlength\belowcaptionskip
\setlength\abovecaptionskip{10\p@}
\setlength\belowcaptionskip{0\p@}

\long\def\@makecaption#1#2{%
	\vskip\abovecaptionskip
	\sbox\@tempboxa{#1 #2}%
	\ifdim \wd\@tempboxa >\hsize
	#1 #2\par
	\else
	\global \@minipagefalse
	\hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
	\fi
	\vskip\belowcaptionskip
}

\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}
\DeclareRobustCommand*\cal{\@fontswitch\relax\mathcal}
\DeclareRobustCommand*\mit{\@fontswitch\relax\mathnormal}
\newcommand\@pnumwidth{1.55em}
\newcommand\@tocrmarg{2.55em}
\newcommand\@dotsep{4.5}
\setcounter{tocdepth}{6}

\newcommand\tableofcontents{%
	\chapter*{\contentsname}
	\@mkboth{\MakeUppercase\contentsname}{\MakeUppercase\contentsname}
	\vspace{.7cm}
	{\hfill \bfseries Page\par}     
	\@starttoc{toc}%
}

\newcommand*\l@chapter[2]{%
  \ifnum \c@tocdepth >\m@ne
	    \addpenalty{-\@highpenalty}%
	    \vskip 1.0em \@plus\p@
	    \setlength\@tempdima{1.5em}%
	    \begingroup
		      \parindent \z@ \rightskip \@pnumwidth
		      \parfillskip -\@pnumwidth
		      \leavevmode \bfseries
		      \hskip -\leftskip
		      #1\nobreak\hfil \nobreak #2 \par
		      \penalty\@highpenalty
	    \endgroup
  \fi
}
  
\newcommand*\l@section{\@dottedtocline{1}{1.5em}{2.3em}}
\newcommand*\l@subsection{\@dottedtocline{2}{3.8em}{3.2em}}
\newcommand*\l@subsubsection{\@dottedtocline{3}{7.0em}{4.1em}}
\newcommand*\l@paragraph{\@dottedtocline{4}{10em}{5em}}
\newcommand*\l@subparagraph{\@dottedtocline{5}{12em}{6em}}

\newcommand\listoffigures{%
	\chapter*{\listfigurename}%
	\@mkboth{\MakeUppercase\listfigurename}%
	{\MakeUppercase\listfigurename}%
	\vspace{.7cm}
	{\hfill \bfseries Page\par}\vspace{-12\p@}
	\addcontentsline{toc}{part}{\MakeUppercase{\listfigurename}}
	\@starttoc{lof}%
}
\newcommand*\l@figure{\@dottedtocline{1}{1.5em}{2.3em}}
\newcommand\listoftables{%
	\chapter*{\listtablename}%
	\@mkboth{%
		\MakeUppercase\listtablename}%
	{\MakeUppercase\listtablename}%
	\vspace{.7cm}
	{\hfill \bfseries Page\par}\vspace{-12\p@}
	\addcontentsline{toc}{part}{\MakeUppercase{\listtablename}}
	\@starttoc{lot}%
	%\clearpage
	%\pagenumbering{arabic}
}
\let\l@table\l@figure
\newdimen\bibindent
\setlength\bibindent{1.5em}
\newenvironment{thebibliography}[1]
     {\chapter*{\bibname}%
      \@mkboth{\MakeUppercase\bibname}{\MakeUppercase\bibname}%
      \list{\@biblabel{\@arabic\c@enumiv}}%
           {\settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth
            \advance\leftmargin\labelsep
            \@openbib@code
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
      \sloppy
      \clubpenalty4000
      \@clubpenalty \clubpenalty
      \widowpenalty4000%
      \sfcode`\.\@m}
     {\def\@noitemerr
       {\@latex@warning{Empty `thebibliography' environment}}%
      \endlist}
\newcommand\newblock{\hskip .11em\@plus.33em\@minus.07em}
\let\@openbib@code\@empty
\newenvironment{theindex}
               {\if@twocolumn
                  \@restonecolfalse
                \else
                  \@restonecoltrue
                \fi
                \twocolumn[\@makeschapterhead{\indexname}]%
                \@mkboth{\MakeUppercase\indexname}%
                        {\MakeUppercase\indexname}%
                \thispagestyle{plain}\parindent\z@
                \parskip\z@ \@plus .3\p@\relax
                \columnseprule \z@
                \columnsep 35\p@
                \let\item\@idxitem}
               {\if@restonecol\onecolumn\else\clearpage\fi}
\newcommand\@idxitem{\par\hangindent 40\p@}
\newcommand\subitem{\@idxitem \hspace*{20\p@}}
\newcommand\subsubitem{\@idxitem \hspace*{30\p@}}
\newcommand\indexspace{\par \vskip 10\p@ \@plus5\p@ \@minus3\p@\relax}
\renewcommand\footnoterule{%
  \kern-3\p@
  \hrule\@width.4\columnwidth
  \kern2.6\p@}
\@addtoreset{footnote}{chapter}
\newcommand\@makefntext[1]{%
    \parindent 1em%
    \noindent
    \hb@xt@1.8em{\hss\@makefnmark}#1}
\newcommand\contentsname{CONTENTS}
\newcommand\listfigurename{LIST OF FIGURES}
\newcommand\listtablename{LIST OF TABLES}
\newcommand\bibname{REFERENCES}
\newcommand\indexname{Index}
\newcommand\figurename{Figure}
\newcommand\tablename{Table}
\newcommand\partname{Part}
\newcommand\chaptername{CHAPTER}
\newcommand\appendixname{APPENDICES}
\newcommand\listofsymbolsname{LIST OF SYMBOLS}
\newcommand\listofabbreviationsname{LIST OF ABBREVIATIONS}
\newcommand\abstractname{Abstract}
\def\today{\ifcase\month\or
	January\or February\or March\or April\or May\or June\or
	July\or August\or September\or October\or November\or December\fi
	\space\number\day, \number\year}
\setlength\columnsep{10\p@}
\setlength\columnseprule{0\p@}
\pagestyle{plain}
\pagenumbering{arabic}
\if@twoside
\else
  \raggedbottom
\fi
\if@twocolumn
  \twocolumn
  \sloppy
  \flushbottom
\else
  \onecolumn
\fi

\def\supervisor#1{\gdef\@supervisor{#1}}

\renewenvironment{abstract}
{	
	\clearpage
	\chapter*{\center \@title \vskip0.05in ABSTRACT}
	\addcontentsline{toc}{part}{\MakeUppercase{ABSTRACT}}	
	\vskip0.2in
}
{
	
}

\def\startappendix{
	\clearpage
	\addcontentsline{toc}{chapter}{\protect\MakeUppercase{\appendixname}\dotdot\nobreak}	
	\appendix
	\chapter*{\appendixname}
	\setcounter{chapter}{0}%
	\setcounter{section}{0}%
	\setcounter{figure}{0}%
	\setcounter{table}{0}%
	\setcounter{equation}{0}
	\renewcommand{\theequation}{A.\arabic{equation}}
	\renewcommand{\thechapter}{A.\arabic{section}}
	\renewcommand{\thesection}{A.\arabic{section}}
	\renewcommand{\theAsection}{Appendix \arabic{section}}
	\renewcommand{\thetable}{Table~A.\@arabic\c@table}
	\renewcommand{\thefigure}{Figure~A.\@arabic\c@figure}
}

\def\startlistofabbreviations{
	\clearpage
	\addcontentsline{toc}{part}{\protect\MakeUppercase{\listofabbreviationsname}\nobreak}	
	\chapter*{\listofabbreviationsname}
}

\def\startlistofsymbols{
	\clearpage
	\addcontentsline{toc}{part}{\protect\MakeUppercase{\listofsymbolsname}\nobreak}	
	\chapter*{\listofsymbolsname}
}

\def \addappendixpart#1{
	\addtocounter{section}{1}
	\addcontentsline{toc}{section}{\thesection :\ #1\protect\leaders\hbox{$\m@th
			\mkern\@dotsep mu\hbox{.}\mkern\@dotsep
			mu$}\hfil}
	\section*{\thesection\protect \; #1}
}

\def\startbibliography{
	\clearpage
	\addcontentsline{toc}{chapter}{\protect\MakeUppercase{\bibname}\nobreak}
	\addtocontents{toc}{\vspace{-1.0em}}
	\bibliographystyle{apalike}
}


\endinput
%%
%% End of file `report.cls'.
