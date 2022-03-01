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
;(setf rows (define fileparser (CSV:parse-file csv_path " \n|\r" "\r")))
 (set 'allrows (read-file csv_path))
 (setf all_splits '())
   (dolist (row (parse allrows "\n" 0)) 
     ;(set 'split (parser row))
     (append-file path row)
    )
)


(write_to "ambience/aft.lsp" "midi/csv.csv")




