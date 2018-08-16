# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 15:16:55 2018

@author: v.shkaberda
"""
from numpy import asarray

import pyodbc

class DBConnect(object):
    ''' Provides connection to database and functions to work with server.
    '''
    def __enter__(self):
        # Connection properties
        conn_str = (
            r'Driver={SQL Server};'
            r'Server=s-kv-center-s64;'
            r'Database=SILPOAnalitic;'
            r'Trusted_Connection=yes;'
            )
        self.__db = pyodbc.connect(conn_str)
        self.__cursor = self.__db.cursor()
        return self


    def __exit__(self, type, value, traceback):
        self.__db.close()


    def connection_check(self):
        self.__cursor.execute('''SELECT SUBSTRING(ORIGINAL_LOGIN(),CHARINDEX('\\',
                                                  ORIGINAL_LOGIN())+1,
                                                  LEN(ORIGINAL_LOGIN()))''')
        return self.__cursor.fetchone()[0]


    def get_data(self, Type, BusinessName):
        ''' Download working data from server.
            Type - int: 2 - Pallet; 3 - m3.
        '''
        self.__cursor.execute('''SELECT FilID, MacroRegionID, Stable,
            [201706], [201707], [201708], [201709], [201710], [201711], [201712],
            [201801], [201802], [201803], [201804], [201805], [201806], [201807]
            FROM SILPOAnalitic.dbo.aid_budget_Filial_Stable_ForClaster
            WHERE EdIzm = ? and BusinessName = ?
            ''', (Type, BusinessName))

        # 'num_rows' needed to reshape the 1D NumPy array returend by 'fromiter'
        # in other words, to restore original dimensions of the results set

        return asarray(self.__cursor.fetchall(), dtype=float)


    def get_id_lists(self, list_name):
        ''' Download id lists from server
        '''
        if list_name == 'regions':
            self.__cursor.execute('SELECT [macroRegionId], [macroRegionNameRu] \
                                  FROM [MasterData].[fil].[ListMacroRegions]')
        elif list_name == 'filials':
            self.__cursor.execute('SELECT [FilID], [FilialName] \
                                  FROM [SILPOAnalitic].[dbo].[aid_FilialsAll]')
        return dict(self.__cursor.fetchall())


if __name__ == '__main__':
    import getpass
    with DBConnect() as dbconn:
        assert dbconn.connection_check() == getpass.getuser(), 'Connection check failed.'
    print('Connected successfully.')
    input('Press any button.')