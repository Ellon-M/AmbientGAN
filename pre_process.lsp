#!/usr/bin/newlisp

;pre-process.lsp

;parses csv rows into lisp lists for any necessary pre-processing.
;writes the processed output as a list of lists to an external file.


(load "csv.lsp")

(define (write_to path csv_path)
  ;#Params:
  	;path: path to destination file
	;csv_path: path to the CSV file containing the data to be processed
		
 (setq parser (CSV:make-row-parser " " "\""))
 (setf rows (define fileparser (CSV:parse-file csv_path "," "\"")))
 (setf all_splits '())
 (dolist (row rows)
   (set 'newrow (slice row 2))
   (setf split_row '())
   (dolist (eachrow newrow) 
     (set 'split (parser eachrow))
     (setf merged_row (push split split_row -1))
    )
   (setf all_merges (push merged_row all_splits -1))
 )

 (write-file path (string all_splits))
)

(write_to "ambience/fore1.lsp" "midi/notes_on.csv")

